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
import { getProviderResolver } from './providers/registry'
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
					content: message.content
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
					...(message.content || []).map((part: AIMessageContentPart) => ({
						type: 'text' as const,
						text: part.type === 'text' ? part.text : JSON.stringify(part),
					})),
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
		throw new Error(`Unsupported AI message role: ${(message as AIMessage).role}`)
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

function toAssistantMessage(
	result: Pick<GenerateTextResult<any, any>, 'text' | 'toolCalls'> & {
		response?: { body?: unknown }
	},
	interleavedField?: string,
): AIMessage {
	const toolCalls = result.toolCalls.map((toolCall) => ({
		id: toolCall.toolCallId,
		type: 'function' as const,
		function: {
			name: toolCall.toolName,
			arguments: JSON.stringify(toolCall.input ?? {}),
		},
	}))

	const message: AIMessage =
		toolCalls.length > 0
			? {
					role: 'assistant',
					content: toTextParts(result.text),
					tool_calls: toolCalls,
				}
			: {
					role: 'assistant',
					content: toTextParts(result.text) || [],
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
	const resolver = getProviderResolver(request.provider)
	const modelName =
		request.provider.models[request.model]?.name?.trim() || request.model
	const interleavedField = getInterleavedMessageField(
		request.provider,
		request.model,
	)
	if (request.onTextDelta && !interleavedField) {
		return streamAssistantTurn(request)
	}

	const { model, providerName } = resolver.createLanguageModel(
		request.provider as never,
		request.model,
		{ messages: request.messages, interleavedField },
	)
	const result = await generateText({
		model,
		messages: toModelMessages(request.messages),
		tools: toAISDKTools(request.tools),
		stopWhen: stepCountIs(1),
		temperature: request.temperature,
		maxOutputTokens: request.maxTokens,
		experimental_include: {
			responseBody: !!interleavedField,
		},
	})

	if (request.onTextDelta && result.text) {
		await request.onTextDelta(result.text, result.text)
	}

	return {
		message: toAssistantMessage(result, interleavedField),
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

	const [text, toolCalls, usage, response] = await Promise.all([
		result.text,
		result.toolCalls,
		result.usage,
		result.response,
	])

	return {
		message: toAssistantMessage(
			{
				text,
				toolCalls,
				response: response as LanguageModelResponseMetadata & { body?: unknown },
			},
			interleavedField,
		),
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
	if (!prompt || (typeof prompt === 'object' && !prompt.text && !prompt.images.length)) {
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