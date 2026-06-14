import type {
	GenerateTextResult,
	ImageModelUsage,
	LanguageModelResponseMetadata,
	LanguageModelUsage,
	ModelMessage,
} from 'ai'
import {
	tool as aiTool,
	generateImage,
	generateText,
	stepCountIs,
	streamText,
} from 'ai'
import { getInterleavedMessageField } from './interleaved-message-field'
import { getOpenAIChatCompletionURLs } from './providers/openai-base-url'
import { getProviderResolver } from './providers/registry'
import { obsidianFetch } from './transport/obsidian-fetch'
import {
	AIMessage,
	AIMessageContentPart,
	AIMessageMeta,
	AIProviderConfig,
	AIToolCall,
	AIToolDefinition,
} from './types'

export interface GenerateAssistantTurnRequest {
	provider: AIProviderConfig
	model: string
	messages: AIMessage[]
	tools: AIToolDefinition[]
	temperature?: number
	maxTokens?: number
	onTextDelta?: (delta: string, fullText: string) => void | Promise<void>
}

interface GenerateTextAssistantTurnOptions {
	disableTools?: boolean
	disableOptionalParams?: boolean
	minimalMessages?: boolean
}

interface DirectStreamRequestInit {
	method: string
	headers: Record<string, string>
	body: string
}

interface NodeRequestModule {
	request: (
		options: Record<string, unknown>,
		callback: (response: NodeIncomingMessage) => void,
	) => NodeClientRequest
}

interface NodeClientRequest {
	on: (event: string, listener: (...args: any[]) => void) => NodeClientRequest
	write: (body: string) => void
	end: () => void
	destroy?: (error?: Error) => void
	setTimeout?: (timeout: number, callback?: () => void) => void
}

interface NodeIncomingMessage {
	statusCode?: number
	statusMessage?: string
	headers?: Record<string, string | string[] | undefined>
	on: (event: string, listener: (...args: any[]) => void) => NodeIncomingMessage
	destroy?: (error?: Error) => void
}

export interface GenerateAssistantTurnResult {
	message: AIMessage
	meta: AIMessageMeta
}

export interface GenerateImageTurnResult {
	contentBase64: string
	mediaType: string
	meta: AIMessageMeta
	prompt: string
}

export const AI_REQUEST_TIMEOUT_MS = 45_000

function withTimeout<T>(
	promise: Promise<T>,
	timeoutMs: number,
	message: string,
): Promise<T> {
	let timeoutId: number | undefined
	const timeout = new Promise<never>((_, reject) => {
		timeoutId = globalThis.setTimeout(() => {
			reject(new Error(message))
		}, timeoutMs) as unknown as number
	})
	return Promise.race([promise, timeout]).finally(() => {
		if (timeoutId !== undefined) {
			globalThis.clearTimeout(timeoutId)
		}
	})
}

function toTextParts(text?: string | null): AIMessageContentPart[] | null {
	if (!text) {
		return null
	}
	return [{ type: 'text', text }]
}

function toModelMessages(messages: AIMessage[]): ModelMessage[] {
	return messages.map((message) => {
		switch (message.role) {
			case 'system':
				return {
					role: 'system',
					content: (message.content || [])
						.filter(
							(
								part: AIMessageContentPart,
							): part is Extract<AIMessageContentPart, { type: 'text' }> =>
								part.type === 'text',
						)
						.map(
							(part: Extract<AIMessageContentPart, { type: 'text' }>) =>
								part.text,
						)
						.join('\n'),
				}
			case 'user': {
				const content = message.content.map((part: AIMessageContentPart) => {
					if (part.type === 'image_url') {
						return {
							type: 'image' as const,
							image: part.image_url.url,
						}
					}
					return {
						type: 'text' as const,
						text: part.type === 'text' ? part.text : JSON.stringify(part.value),
					}
				})
				return {
					role: 'user',
					content,
				}
			}
			case 'assistant': {
				const content = [
					...(message.content || []).flatMap((part: AIMessageContentPart) => {
						if (part.type === 'image_url') {
							return []
						}
						return [
							{
								type: 'text' as const,
								text:
									part.type === 'text' ? part.text : JSON.stringify(part.value),
							},
						]
					}),
					...(message.tool_calls || []).map((toolCall: AIToolCall) => ({
						type: 'tool-call' as const,
						toolCallId: toolCall.id,
						toolName: toolCall.function.name,
						input: JSON.parse(toolCall.function.arguments || '{}'),
					})),
				]
				return {
					role: 'assistant',
					content,
				}
			}
			case 'tool':
				return {
					role: 'tool',
					content: [
						{
							type: 'tool-result' as const,
							toolCallId: message.tool_call_id,
							toolName: message.name,
							output: {
								type: 'text' as const,
								value: message.content
									.filter(
										(
											part: AIMessageContentPart,
										): part is Extract<
											AIMessageContentPart,
											{ type: 'text' }
										> => part.type === 'text',
									)
									.map(
										(part: Extract<AIMessageContentPart, { type: 'text' }>) =>
											part.text,
									)
									.join('\n'),
							},
						},
					],
				}
		}
		throw new Error(
			`Unsupported AI message role: ${(message as AIMessage).role}`,
		)
	})
}

function toAISDKTools(tools: AIToolDefinition[]) {
	return Object.fromEntries(
		tools.map((toolDefinition) => [
			toolDefinition.name,
			aiTool({
				description: toolDefinition.description,
				inputSchema: toolDefinition.inputSchema,
			}),
		]),
	)
}

function dataUrlFromBase64(value: string, mediaType = 'image/png') {
	return value.startsWith('data:') ? value : `data:${mediaType};base64,${value}`
}

function contentPartFromRawPart(part: unknown): AIMessageContentPart[] {
	if (typeof part === 'string') {
		return toTextParts(part) || []
	}
	if (!part || typeof part !== 'object') {
		return []
	}
	const value = part as Record<string, any>
	const text =
		typeof value.text === 'string'
			? value.text
			: typeof value.output_text === 'string'
				? value.output_text
				: undefined
	const imageUrl =
		typeof value.image_url === 'string'
			? value.image_url
			: typeof value.image_url?.url === 'string'
				? value.image_url.url
				: typeof value.url === 'string' && value.type?.includes?.('image')
					? value.url
					: undefined
	const base64 =
		typeof value.b64_json === 'string'
			? value.b64_json
			: typeof value.base64 === 'string'
				? value.base64
				: undefined

	if (text) {
		return [{ type: 'text', text }]
	}
	if (imageUrl) {
		return [{ type: 'image_url', image_url: { url: imageUrl } }]
	}
	if (base64) {
		return [
			{
				type: 'image_url',
				image_url: {
					url: dataUrlFromBase64(base64, value.mediaType || value.media_type),
				},
			},
		]
	}
	if (Array.isArray(value.content)) {
		return value.content.flatMap(contentPartFromRawPart)
	}
	return []
}

function contentPartsFromRawContent(content: unknown): AIMessageContentPart[] {
	if (Array.isArray(content)) {
		return content.flatMap(contentPartFromRawPart)
	}
	return contentPartFromRawPart(content)
}

function extractRawAssistantContentParts(
	body: unknown,
): AIMessageContentPart[] {
	if (!body || typeof body !== 'object') {
		return []
	}
	const value = body as Record<string, any>
	const choiceParts = Array.isArray(value.choices)
		? value.choices.flatMap((choice: any) =>
				contentPartsFromRawContent(choice?.message?.content),
			)
		: []
	if (choiceParts.length > 0) {
		return choiceParts
	}
	if (typeof value.output_text === 'string') {
		return toTextParts(value.output_text) || []
	}
	if (Array.isArray(value.output)) {
		return value.output.flatMap((item: any) =>
			contentPartsFromRawContent(item?.content ?? item),
		)
	}
	return contentPartsFromRawContent(value.content)
}

function toAssistantMessage(
	result: Pick<GenerateTextResult<any, any>, 'text' | 'toolCalls'> & {
		files?: Array<{ base64: string; mediaType: string }>
		response?: { body?: unknown }
	},
	interleavedField?: string,
): AIMessage {
	const toolCalls = (result.toolCalls || []).map((toolCall) => ({
		id: toolCall.toolCallId,
		type: 'function' as const,
		function: {
			name: toolCall.toolName,
			arguments: JSON.stringify(toolCall.input ?? {}),
		},
	}))
	const imageParts = (result.files || [])
		.filter((file) => file.mediaType?.startsWith('image/'))
		.map((file) => ({
			type: 'image_url' as const,
			image_url: {
				url: file.base64.startsWith('data:')
					? file.base64
					: `data:${file.mediaType};base64,${file.base64}`,
			},
		}))
	const rawParts = result.text
		? []
		: extractRawAssistantContentParts(result.response?.body)
	const usableRawParts = imageParts.length
		? rawParts.filter((part) => part.type === 'text')
		: rawParts
	const content = [
		...(toTextParts(result.text) || []),
		...usableRawParts,
		...imageParts,
	]

	const message: AIMessage =
		toolCalls.length > 0
			? {
					role: 'assistant',
					content: content.length ? content : null,
					tool_calls: toolCalls,
				}
			: {
					role: 'assistant',
					content,
				}

	if (interleavedField && message.role === 'assistant') {
		const body = result.response?.body as any
		const raw = body?.choices?.[0]?.message?.[interleavedField]
		if (raw !== undefined) {
			message.interleaved = { [interleavedField]: raw }
		}
	}

	return message
}

function hasAssistantOutput(message: AIMessage) {
	if (message.role !== 'assistant') {
		return true
	}
	const hasContent = (message.content || []).some((part) => {
		if (part.type === 'text') {
			return part.text.trim().length > 0
		}
		return true
	})
	return hasContent || !!message.tool_calls?.length
}

function assertAssistantOutput(message: AIMessage) {
	if (!hasAssistantOutput(message)) {
		throw new Error('No assistant output was generated.')
	}
}

function toMeta(params: {
	provider: AIProviderConfig
	providerName: string
	modelName: string
	usage?: LanguageModelUsage
}) {
	return {
		providerId: params.provider.id,
		providerName: params.provider.name || params.providerName,
		modelName: params.modelName,
		usage: {
			inputTokens: params.usage?.inputTokens,
			outputTokens: params.usage?.outputTokens,
			totalTokens: params.usage?.totalTokens,
		},
	} satisfies AIMessageMeta
}

function toImageMeta(params: {
	provider: AIProviderConfig
	providerName: string
	modelName: string
	usage?: ImageModelUsage
}) {
	const usage = params.usage as
		| {
				inputTokens?: number
				outputTokens?: number
				totalTokens?: number
				tokens?: number
		  }
		| undefined

	return {
		providerId: params.provider.id,
		providerName: params.provider.name || params.providerName,
		modelName: params.modelName,
		usage: {
			inputTokens: usage?.inputTokens ?? usage?.tokens,
			outputTokens: usage?.outputTokens,
			totalTokens: usage?.totalTokens ?? usage?.tokens,
		},
	} satisfies AIMessageMeta
}

function getLastUserMessage(messages: AIMessage[]) {
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index]
		if (message.role === 'user') {
			return message
		}
	}
	return undefined
}

function getLastMessage(messages: AIMessage[]) {
	return messages[messages.length - 1]
}

function hasImageParts(message: AIMessage | undefined) {
	return !!message?.content?.some((part) => part.type === 'image_url')
}

function shouldTryDirectTextFirst(
	request: GenerateAssistantTurnRequest,
	interleavedField?: string,
) {
	const lastMessage = getLastMessage(request.messages)
	return (
		!interleavedField &&
		lastMessage?.role === 'user' &&
		!hasImageParts(lastMessage)
	)
}

function toMinimalMessages(messages: AIMessage[]) {
	const lastUserMessage = getLastUserMessage(messages)
	return lastUserMessage ? [lastUserMessage] : messages
}

function toImagePrompt(message: AIMessage | undefined) {
	if (!message) {
		return ''
	}

	const text = (message.content || [])
		.filter(
			(part): part is Extract<AIMessageContentPart, { type: 'text' }> =>
				part.type === 'text',
		)
		.map((part) => part.text)
		.join('\n')
		.trim()

	const images = (message.content || [])
		.filter(
			(part): part is Extract<AIMessageContentPart, { type: 'image_url' }> =>
				part.type === 'image_url',
		)
		.map((part) => part.image_url.url)

	return images.length
		? {
				text,
				images,
			}
		: text
}

export function assertProviderUsable(provider: AIProviderConfig) {
	getProviderResolver(provider).assertUsable(provider)
}

export async function generateAssistantTurn(
	request: GenerateAssistantTurnRequest,
): Promise<GenerateAssistantTurnResult> {
	const interleavedField = getInterleavedMessageField(
		request.provider,
		request.model,
	)
	if (request.onTextDelta && !interleavedField) {
		if (shouldTryDirectTextFirst(request, interleavedField)) {
			try {
				return await withTimeout(
					streamTextAssistantTurnDirect(request),
					AI_REQUEST_TIMEOUT_MS,
					'Direct streaming request timed out; trying SDK streaming fallback.',
				)
			} catch {
				// Fall through to SDK streaming and compatibility fallbacks.
			}
		}
		try {
			return await withTimeout(
				streamAssistantTurn(request),
				AI_REQUEST_TIMEOUT_MS,
				'Streaming request timed out; trying non-streaming fallback.',
			)
		} catch {
			return generateTextAssistantTurnWithFallbacks(request, interleavedField)
		}
	}

	if (shouldTryDirectTextFirst(request, interleavedField)) {
		try {
			return await generateTextAssistantTurnDirect(request)
		} catch {
			// Fall through to SDK compatibility fallbacks.
		}
	}

	return generateTextAssistantTurnWithFallbacks(request, interleavedField)
}

async function generateTextAssistantTurnWithFallbacks(
	request: GenerateAssistantTurnRequest,
	interleavedField?: string,
): Promise<GenerateAssistantTurnResult> {
	const attempts: GenerateTextAssistantTurnOptions[] = [
		{},
		{ disableTools: true },
		{ disableTools: true, disableOptionalParams: true },
		{
			disableTools: true,
			disableOptionalParams: true,
			minimalMessages: true,
		},
	]
	let lastError: unknown
	for (const options of attempts) {
		try {
			return await withTimeout(
				generateTextAssistantTurn(request, interleavedField, options),
				AI_REQUEST_TIMEOUT_MS,
				'Text request timed out; trying safer fallback.',
			)
		} catch (error) {
			lastError = error
		}
	}
	try {
		return await generateTextAssistantTurnDirect(request)
	} catch (error) {
		lastError = error
	}
	throw lastError instanceof Error
		? lastError
		: new Error('No assistant output was generated.')
}

function toOpenAITextMessage(message: AIMessage) {
	const content = (message.content || [])
		.filter(
			(part): part is Extract<AIMessageContentPart, { type: 'text' }> =>
				part.type === 'text',
		)
		.map((part) => part.text)
		.join('\n')
	return {
		role: message.role,
		content,
	}
}

function createDirectTextMessages(messages: AIMessage[]) {
	const lastUserMessage = getLastUserMessage(messages)
	return (lastUserMessage ? [lastUserMessage] : messages)
		.filter((message) => message.role === 'user' || message.role === 'system')
		.map(toOpenAITextMessage)
		.filter((message) => message.content.trim().length > 0)
}

function createDirectTextRequestBody(
	request: GenerateAssistantTurnRequest,
	messages: ReturnType<typeof createDirectTextMessages>,
	stream: boolean,
) {
	const body: Record<string, unknown> = {
		model: request.model,
		messages,
		stream,
	}
	if (
		typeof request.temperature === 'number' &&
		Number.isFinite(request.temperature)
	) {
		body.temperature = request.temperature
	}
	if (
		typeof request.maxTokens === 'number' &&
		Number.isFinite(request.maxTokens)
	) {
		body.max_tokens = request.maxTokens
	}
	return body
}

async function generateTextAssistantTurnDirect(
	request: GenerateAssistantTurnRequest,
): Promise<GenerateAssistantTurnResult> {
	const messages = createDirectTextMessages(request.messages)
	if (messages.length === 0) {
		throw new Error('Enter a message before generating a response.')
	}

	const modelName =
		request.provider.models[request.model]?.name?.trim() || request.model
	let lastError: unknown
	for (const url of getOpenAIChatCompletionURLs(request.provider.api)) {
		try {
			const response = await withTimeout(
				obsidianFetch(url, {
					method: 'POST',
					headers: {
						'content-type': 'application/json',
						authorization: `Bearer ${request.provider.apiKey}`,
					},
					body: JSON.stringify(
						createDirectTextRequestBody(request, messages, false),
					),
				}),
				AI_REQUEST_TIMEOUT_MS,
				`Direct text request timed out: ${url}`,
			)
			if (!response) {
				throw new Error(`Direct text request failed: ${url}`)
			}
			const body = await response.json().catch(() => undefined)
			if (!response.ok) {
				throw new Error(
					typeof body?.error?.message === 'string'
						? `${body.error.message} (${url})`
						: `Direct text request failed: ${response.status} ${response.statusText}`,
				)
			}
			const message = toAssistantMessage({
				text: '',
				toolCalls: [],
				files: [],
				response: { body },
			})
			assertAssistantOutput(message)
			if (request.onTextDelta) {
				const text = (message.content || [])
					.filter(
						(part): part is Extract<AIMessageContentPart, { type: 'text' }> =>
							part.type === 'text',
					)
					.map((part) => part.text)
					.join('\n')
				if (text) {
					await request.onTextDelta(text, text)
				}
			}
			return {
				message,
				meta: {
					providerId: request.provider.id,
					providerName: request.provider.name || 'OpenAI',
					modelName,
					usage: body?.usage
						? {
								inputTokens: body?.usage?.prompt_tokens,
								outputTokens: body?.usage?.completion_tokens,
								totalTokens: body?.usage?.total_tokens,
							}
						: undefined,
				},
			}
		} catch (error) {
			lastError = error
		}
	}
	throw lastError instanceof Error
		? lastError
		: new Error('Direct text request failed.')
}

function getRuntimeRequire(): ((moduleName: string) => unknown) | undefined {
	const candidate =
		(globalThis as any).require || (globalThis as any).window?.require
	return typeof candidate === 'function' ? candidate : undefined
}

function getNodeRequestModule(target: URL): NodeRequestModule | undefined {
	const runtimeRequire = getRuntimeRequire()
	if (!runtimeRequire) {
		return undefined
	}
	try {
		const moduleName = target.protocol === 'https:' ? 'https' : 'http'
		const transport = runtimeRequire(moduleName) as Partial<NodeRequestModule>
		return typeof transport?.request === 'function'
			? (transport as NodeRequestModule)
			: undefined
	} catch {
		return undefined
	}
}

function toUint8Array(chunk: unknown) {
	if (chunk instanceof Uint8Array) {
		return chunk
	}
	if (chunk instanceof ArrayBuffer) {
		return new Uint8Array(chunk)
	}
	if (ArrayBuffer.isView(chunk)) {
		const view = chunk as ArrayBufferView
		return new Uint8Array(view.buffer, view.byteOffset, view.byteLength)
	}
	return new TextEncoder().encode(String(chunk ?? ''))
}

function appendNodeResponseHeaders(
	headers: Headers,
	source?: Record<string, string | string[] | undefined>,
) {
	for (const [key, value] of Object.entries(source || {})) {
		if (Array.isArray(value)) {
			for (const item of value) {
				headers.append(key, item)
			}
		} else if (value !== undefined) {
			headers.set(key, value)
		}
	}
}

function createNodeStreamingResponse(
	url: string,
	init: DirectStreamRequestInit,
	transport: NodeRequestModule,
): Promise<Response> {
	const target = new URL(url)
	const headers = {
		...init.headers,
		'content-length': String(new TextEncoder().encode(init.body).byteLength),
	}

	return new Promise((resolve, reject) => {
		let settled = false
		let nodeRequest: NodeClientRequest | undefined
		const finishWithError = (error: unknown) => {
			if (settled) {
				return
			}
			settled = true
			reject(error)
		}

		nodeRequest = transport.request(
			{
				protocol: target.protocol,
				hostname: target.hostname,
				port: target.port || undefined,
				path: `${target.pathname}${target.search}`,
				method: init.method,
				headers,
			},
			(nodeResponse) => {
				const responseHeaders = new Headers()
				appendNodeResponseHeaders(responseHeaders, nodeResponse.headers)
				const body = new ReadableStream<Uint8Array>({
					start(controller) {
						nodeResponse.on('data', (chunk) => {
							controller.enqueue(toUint8Array(chunk))
						})
						nodeResponse.on('end', () => {
							controller.close()
						})
						nodeResponse.on('error', (error) => {
							controller.error(error)
						})
					},
					cancel() {
						nodeResponse.destroy?.()
						nodeRequest?.destroy?.()
					},
				})

				settled = true
				resolve(
					new Response(body, {
						status: nodeResponse.statusCode || 200,
						statusText: nodeResponse.statusMessage || '',
						headers: responseHeaders,
					}),
				)
			},
		)
		nodeRequest.on('error', finishWithError)
		nodeRequest.setTimeout?.(AI_REQUEST_TIMEOUT_MS, () => {
			nodeRequest?.destroy?.(new Error('Direct streaming request timed out.'))
		})
		nodeRequest.write(init.body)
		nodeRequest.end()
	})
}

async function createDirectStreamingResponse(
	url: string,
	init: DirectStreamRequestInit,
) {
	const target = new URL(url)
	const nodeTransport = getNodeRequestModule(target)
	if (nodeTransport) {
		return createNodeStreamingResponse(url, init, nodeTransport)
	}

	const nativeFetch = globalThis.fetch?.bind(globalThis)
	if (!nativeFetch) {
		throw new Error('Native streaming transport is unavailable.')
	}
	return nativeFetch(url, {
		method: init.method,
		headers: init.headers,
		body: init.body,
	})
}

async function* readTextChunks(body: ReadableStream<Uint8Array>) {
	const reader = body.getReader()
	const decoder = new TextDecoder()
	try {
		while (true) {
			const { done, value } = await reader.read()
			if (done) {
				break
			}
			if (value) {
				yield decoder.decode(value, { stream: true })
			}
		}
		const tail = decoder.decode()
		if (tail) {
			yield tail
		}
	} finally {
		reader.releaseLock()
	}
}

function extractTextFromDirectStreamPayload(payload: string) {
	if (!payload || payload === '[DONE]') {
		return { done: payload === '[DONE]', text: '' }
	}
	try {
		const body = JSON.parse(payload)
		const usage = body?.usage
		let text = ''
		if (typeof body?.delta === 'string') {
			text += body.delta
		}
		if (typeof body?.output_text === 'string') {
			text += body.output_text
		}
		if (Array.isArray(body?.choices)) {
			for (const choice of body.choices) {
				const delta = choice?.delta ?? choice?.message ?? choice
				if (typeof delta?.content === 'string') {
					text += delta.content
				} else if (delta?.content !== undefined) {
					text += contentPartsFromRawContent(delta.content)
						.filter(
							(part): part is Extract<AIMessageContentPart, { type: 'text' }> =>
								part.type === 'text',
						)
						.map((part) => part.text)
						.join('')
				}
				if (typeof choice?.text === 'string') {
					text += choice.text
				}
			}
		}
		return { done: false, text, usage }
	} catch {
		return { done: false, text: '' }
	}
}

async function streamTextAssistantTurnDirect(
	request: GenerateAssistantTurnRequest,
): Promise<GenerateAssistantTurnResult> {
	const messages = createDirectTextMessages(request.messages)
	if (messages.length === 0) {
		throw new Error('Enter a message before generating a response.')
	}

	const modelName =
		request.provider.models[request.model]?.name?.trim() || request.model
	let lastError: unknown
	for (const url of getOpenAIChatCompletionURLs(request.provider.api)) {
		try {
			const response = await createDirectStreamingResponse(url, {
				method: 'POST',
				headers: {
					'content-type': 'application/json',
					accept: 'text/event-stream',
					authorization: `Bearer ${request.provider.apiKey}`,
				},
				body: JSON.stringify(
					createDirectTextRequestBody(request, messages, true),
				),
			})
			if (!response.ok) {
				const body = await response.json().catch(() => undefined)
				throw new Error(
					typeof body?.error?.message === 'string'
						? `${body.error.message} (${url})`
						: `Direct streaming request failed: ${response.status} ${response.statusText}`,
				)
			}
			if (!response.body) {
				throw new Error('Direct streaming response did not include a body.')
			}

			let buffer = ''
			let fullText = ''
			let usage:
				| {
						prompt_tokens?: number
						completion_tokens?: number
						total_tokens?: number
				  }
				| undefined
			for await (const chunk of readTextChunks(response.body)) {
				buffer += chunk
				const events = buffer.split(/\r?\n\r?\n/)
				buffer = events.pop() || ''
				for (const event of events) {
					for (const line of event.split(/\r?\n/)) {
						const trimmed = line.trim()
						if (!trimmed.startsWith('data:')) {
							continue
						}
						const parsed = extractTextFromDirectStreamPayload(
							trimmed.slice('data:'.length).trim(),
						)
						if (parsed.usage) {
							usage = parsed.usage
						}
						if (parsed.text) {
							fullText += parsed.text
							await request.onTextDelta?.(parsed.text, fullText)
						}
					}
				}
			}
			if (buffer.trim()) {
				for (const line of buffer.split(/\r?\n/)) {
					const trimmed = line.trim()
					if (!trimmed.startsWith('data:')) {
						continue
					}
					const parsed = extractTextFromDirectStreamPayload(
						trimmed.slice('data:'.length).trim(),
					)
					if (parsed.usage) {
						usage = parsed.usage
					}
					if (parsed.text) {
						fullText += parsed.text
						await request.onTextDelta?.(parsed.text, fullText)
					}
				}
			}

			const message: AIMessage = {
				role: 'assistant',
				content: toTextParts(fullText) || [],
			}
			assertAssistantOutput(message)
			return {
				message,
				meta: {
					providerId: request.provider.id,
					providerName: request.provider.name || 'OpenAI',
					modelName,
					usage: usage
						? {
								inputTokens: usage.prompt_tokens,
								outputTokens: usage.completion_tokens,
								totalTokens: usage.total_tokens,
							}
						: undefined,
				},
			}
		} catch (error) {
			lastError = error
		}
	}
	throw lastError instanceof Error
		? lastError
		: new Error('Direct streaming request failed.')
}

async function generateTextAssistantTurn(
	request: GenerateAssistantTurnRequest,
	interleavedField?: string,
	options: GenerateTextAssistantTurnOptions = {},
): Promise<GenerateAssistantTurnResult> {
	const resolver = getProviderResolver(request.provider)
	const modelName =
		request.provider.models[request.model]?.name?.trim() || request.model
	const { model, providerName } = resolver.createLanguageModel(
		request.provider as never,
		request.model,
		{ messages: request.messages, interleavedField },
	)
	const messages = options.minimalMessages
		? toMinimalMessages(request.messages)
		: request.messages
	const result = await generateText({
		model,
		messages: toModelMessages(messages),
		tools: options.disableTools ? {} : toAISDKTools(request.tools),
		stopWhen: stepCountIs(1),
		...(options.disableOptionalParams
			? {}
			: {
					temperature: request.temperature,
					maxOutputTokens: request.maxTokens,
				}),
		experimental_include: {
			responseBody: true,
		},
	})

	if (request.onTextDelta && result.text) {
		await request.onTextDelta(result.text, result.text)
	}
	const message = toAssistantMessage(result, interleavedField)
	assertAssistantOutput(message)

	return {
		message,
		meta: toMeta({
			provider: request.provider,
			providerName,
			modelName,
			usage: result.usage,
		}),
	}
}

export async function streamAssistantTurn(
	request: GenerateAssistantTurnRequest,
): Promise<GenerateAssistantTurnResult> {
	const resolver = getProviderResolver(request.provider)
	const modelName =
		request.provider.models[request.model]?.name?.trim() || request.model
	const interleavedField = getInterleavedMessageField(
		request.provider,
		request.model,
	)
	const { model, providerName } = resolver.createLanguageModel(
		request.provider as never,
		request.model,
		{ messages: request.messages, interleavedField },
	)
	const result = streamText({
		model,
		messages: toModelMessages(request.messages),
		tools: toAISDKTools(request.tools),
		stopWhen: stepCountIs(1),
		temperature: request.temperature,
		maxOutputTokens: request.maxTokens,
	})

	let fullText = ''
	for await (const delta of result.textStream) {
		fullText += delta
		await request.onTextDelta?.(delta, fullText)
	}

	const [text, toolCalls, usage, response, files] = await Promise.all([
		result.text,
		result.toolCalls,
		result.usage,
		result.response,
		result.files,
	])

	const message = toAssistantMessage(
		{
			text,
			toolCalls,
			files,
			response: response as LanguageModelResponseMetadata & { body?: unknown },
		},
		interleavedField,
	)
	assertAssistantOutput(message)

	return {
		message,
		meta: toMeta({
			provider: request.provider,
			providerName,
			modelName,
			usage,
		}),
	}
}

export async function generateImageTurn(
	request: Omit<GenerateAssistantTurnRequest, 'tools' | 'onTextDelta'>,
): Promise<GenerateImageTurnResult> {
	const resolver = getProviderResolver(request.provider)
	if (!resolver.createImageModel) {
		throw new Error('The selected provider does not support image generation.')
	}

	const modelName =
		request.provider.models[request.model]?.name?.trim() || request.model
	const { model, providerName } = resolver.createImageModel(
		request.provider as never,
		request.model,
	)
	const lastUserMessage = getLastUserMessage(request.messages)
	const prompt = toImagePrompt(lastUserMessage)
	if (
		!prompt ||
		(typeof prompt === 'object' && !prompt.text && !prompt.images.length)
	) {
		throw new Error('Enter an image prompt before generating an image.')
	}

	const result = await generateImage({
		model,
		prompt,
		n: 1,
	})
	const image = result.image

	return {
		contentBase64: image.base64,
		mediaType: image.mediaType || 'image/png',
		prompt: typeof prompt === 'string' ? prompt : prompt.text || '',
		meta: toImageMeta({
			provider: request.provider,
			providerName,
			modelName,
			usage: result.usage,
		}),
	}
}