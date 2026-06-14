import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
	AI_REQUEST_TIMEOUT_MS,
	generateAssistantTurn,
	generateImageTurn,
} from './runtime'
import type { AIProviderConfig } from './types'

const aiMocks = vi.hoisted(() => ({
	generateText: vi.fn(),
	streamText: vi.fn(),
	generateImage: vi.fn(),
	stepCountIs: vi.fn((count: number) => ({ count })),
	tool: vi.fn((definition: unknown) => definition),
}))

const providerMocks = vi.hoisted(() => ({
	createLanguageModel: vi.fn(() => ({
		model: { modelId: 'model-1' },
		providerName: 'Provider',
	})),
	createImageModel: vi.fn(() => ({
		model: { modelId: 'gpt-image-2' },
		providerName: 'Provider',
	})),
	assertUsable: vi.fn(),
}))

const transportMocks = vi.hoisted(() => ({
	obsidianFetch: vi.fn(),
}))

vi.mock('ai', () => aiMocks)

vi.mock('~/ai/providers/registry', () => ({
	getProviderResolver: () => providerMocks,
}))

vi.mock('~/ai/interleaved-message-field', () => ({
	getInterleavedMessageField: () => undefined,
}))

vi.mock('~/ai/transport/obsidian-fetch', () => transportMocks)

function createProvider(): AIProviderConfig {
	return {
		id: 'provider-1',
		env: [],
		npm: '@ai-sdk/openai',
		name: 'Provider',
		doc: 'https://example.com',
		api: 'https://example.com/v1',
		apiKey: 'key',
		models: {
			'model-1': {
				id: 'model-1',
				name: 'model-a',
				attachment: false,
				reasoning: false,
				tool_call: true,
				release_date: '2026-01-01',
				last_updated: '2026-01-01',
				modalities: { input: ['text'], output: ['text'] },
				open_weights: false,
				limit: { context: 1000, output: 1000 },
			},
		},
	}
}

describe('generateAssistantTurn', () => {
	beforeEach(() => {
		vi.useRealTimers()
		aiMocks.generateText.mockReset()
		aiMocks.streamText.mockReset()
		aiMocks.generateImage.mockReset()
		aiMocks.stepCountIs.mockClear()
		aiMocks.tool.mockClear()
		providerMocks.createLanguageModel.mockClear()
		providerMocks.createImageModel.mockClear()
		providerMocks.assertUsable.mockClear()
		transportMocks.obsidianFetch.mockReset()
	})

	it('falls back to non-streaming text generation when the stream fails', async () => {
		aiMocks.streamText.mockReturnValueOnce({
			textStream: (async function* () {
				throw new Error('stream failed')
			})(),
		})
		aiMocks.generateText.mockResolvedValueOnce({
			text: 'Fallback answer',
			toolCalls: [],
			usage: {
				inputTokens: 3,
				outputTokens: 2,
				totalTokens: 5,
			},
			response: {},
		})
		const updates: { delta: string; fullText: string }[] = []

		const result = await generateAssistantTurn({
			provider: createProvider(),
			model: 'model-1',
			messages: [
				{
					role: 'user',
					content: [{ type: 'text', text: 'Hello' }],
				},
			],
			tools: [],
			onTextDelta: async (delta, fullText) => {
				updates.push({ delta, fullText })
			},
		})

		expect(aiMocks.streamText).toHaveBeenCalledTimes(1)
		expect(aiMocks.generateText).toHaveBeenCalledTimes(1)
		expect(updates).toEqual([
			{ delta: 'Fallback answer', fullText: 'Fallback answer' },
		])
		expect(result.message).toEqual({
			role: 'assistant',
			content: [{ type: 'text', text: 'Fallback answer' }],
		})
		expect(result.meta).toMatchObject({
			providerId: 'provider-1',
			providerName: 'Provider',
			modelName: 'model-a',
			usage: {
				inputTokens: 3,
				outputTokens: 2,
				totalTokens: 5,
			},
			})
		})

	it('falls back when a streaming response never completes', async () => {
		vi.useFakeTimers()
		aiMocks.streamText.mockReturnValueOnce({
			textStream: (async function* () {
				await new Promise(() => undefined)
			})(),
		})
		aiMocks.generateText.mockResolvedValueOnce({
			text: 'Recovered after stream timeout',
			toolCalls: [],
			usage: {
				inputTokens: 3,
				outputTokens: 2,
				totalTokens: 5,
			},
			response: {},
		})
		const updates: { delta: string; fullText: string }[] = []

		const promise = generateAssistantTurn({
			provider: createProvider(),
			model: 'model-1',
			messages: [
				{
					role: 'user',
					content: [{ type: 'text', text: 'Hello' }],
				},
			],
			tools: [],
			onTextDelta: async (delta, fullText) => {
				updates.push({ delta, fullText })
			},
		})
		await vi.advanceTimersByTimeAsync(AI_REQUEST_TIMEOUT_MS)
		const result = await promise
		vi.useRealTimers()

		expect(aiMocks.streamText).toHaveBeenCalledTimes(1)
		expect(aiMocks.generateText).toHaveBeenCalledTimes(1)
		expect(updates).toEqual([
			{
				delta: 'Recovered after stream timeout',
				fullText: 'Recovered after stream timeout',
			},
		])
		expect(result.message).toEqual({
			role: 'assistant',
			content: [{ type: 'text', text: 'Recovered after stream timeout' }],
		})
	})

	it('retries without tools and optional params when compatible providers reject them', async () => {
		aiMocks.generateText
			.mockRejectedValueOnce(new Error('tools not supported'))
			.mockRejectedValueOnce(new Error('temperature not supported'))
			.mockResolvedValueOnce({
				text: 'Plain fallback answer',
				toolCalls: [],
				files: [],
				usage: {
					inputTokens: 2,
					outputTokens: 3,
					totalTokens: 5,
				},
				response: {},
			})

		const result = await generateAssistantTurn({
			provider: createProvider(),
			model: 'model-1',
			messages: [
				{
					role: 'user',
					content: [{ type: 'text', text: 'Hello' }],
				},
			],
			tools: [
				{
					name: 'search',
					description: 'Search',
					inputSchema: {} as never,
					execute: vi.fn(),
				},
			],
			temperature: 0.7,
			maxTokens: 1200,
		})

		expect(aiMocks.generateText).toHaveBeenCalledTimes(3)
		expect(aiMocks.generateText.mock.calls[0][0]).toMatchObject({
			temperature: 0.7,
			maxOutputTokens: 1200,
			experimental_include: { responseBody: true },
		})
		expect(Object.keys(aiMocks.generateText.mock.calls[0][0].tools)).toEqual([
			'search',
		])
		expect(aiMocks.generateText.mock.calls[1][0].tools).toEqual({})
		expect(aiMocks.generateText.mock.calls[1][0]).toMatchObject({
			temperature: 0.7,
			maxOutputTokens: 1200,
		})
		expect(aiMocks.generateText.mock.calls[2][0].tools).toEqual({})
		expect(aiMocks.generateText.mock.calls[2][0]).not.toHaveProperty(
			'temperature',
		)
		expect(aiMocks.generateText.mock.calls[2][0]).not.toHaveProperty(
			'maxOutputTokens',
		)
		expect(result.message).toEqual({
			role: 'assistant',
			content: [{ type: 'text', text: 'Plain fallback answer' }],
		})
	})

	it('falls back to only the latest user message when full context is rejected', async () => {
		aiMocks.generateText
			.mockRejectedValueOnce(new Error('full context rejected'))
			.mockRejectedValueOnce(new Error('tools rejected'))
			.mockRejectedValueOnce(new Error('params rejected'))
			.mockResolvedValueOnce({
				text: 'Minimal fallback answer',
				toolCalls: [],
				files: [],
				usage: {
					inputTokens: 1,
					outputTokens: 2,
					totalTokens: 3,
				},
				response: {},
			})

		const result = await generateAssistantTurn({
			provider: createProvider(),
			model: 'model-1',
			messages: [
				{
					role: 'system',
					content: [{ type: 'text', text: 'You are helpful.' }],
				},
				{
					role: 'user',
					content: [{ type: 'text', text: 'Old question' }],
				},
				{
					role: 'assistant',
					content: [{ type: 'text', text: 'Old answer' }],
				},
				{
					role: 'user',
					content: [{ type: 'text', text: 'Latest question' }],
				},
			],
			tools: [
				{
					name: 'search',
					description: 'Search',
					inputSchema: {} as never,
					execute: vi.fn(),
				},
			],
			temperature: 0.7,
			maxTokens: 1200,
		})

		expect(aiMocks.generateText).toHaveBeenCalledTimes(4)
		expect(aiMocks.generateText.mock.calls[3][0].tools).toEqual({})
		expect(aiMocks.generateText.mock.calls[3][0]).not.toHaveProperty(
			'temperature',
		)
		expect(aiMocks.generateText.mock.calls[3][0]).not.toHaveProperty(
			'maxOutputTokens',
		)
		expect(aiMocks.generateText.mock.calls[3][0].messages).toEqual([
			{
				role: 'user',
				content: [{ type: 'text', text: 'Latest question' }],
			},
		])
		expect(result.message).toEqual({
			role: 'assistant',
			content: [{ type: 'text', text: 'Minimal fallback answer' }],
		})
	})

	it('falls back to a direct OpenAI-compatible request when SDK attempts fail', async () => {
		aiMocks.generateText
			.mockRejectedValueOnce(new Error('full context rejected'))
			.mockRejectedValueOnce(new Error('tools rejected'))
			.mockRejectedValueOnce(new Error('params rejected'))
			.mockRejectedValueOnce(new Error('minimal sdk rejected'))
		transportMocks.obsidianFetch.mockResolvedValueOnce(
			new Response(
				JSON.stringify({
					choices: [
						{
							message: {
								content: 'Direct fallback answer',
							},
						},
					],
					usage: {
						prompt_tokens: 4,
						completion_tokens: 3,
						total_tokens: 7,
					},
				}),
				{
					status: 200,
					headers: { 'content-type': 'application/json' },
				},
			),
		)
		const updates: { delta: string; fullText: string }[] = []

		const provider = createProvider()
		provider.api = 'https://example.com'
		const result = await generateAssistantTurn({
			provider,
			model: 'model-1',
			messages: [
				{
					role: 'user',
					content: [{ type: 'text', text: 'Old question' }],
				},
				{
					role: 'assistant',
					content: [{ type: 'text', text: 'Old answer' }],
				},
				{
					role: 'user',
					content: [{ type: 'text', text: 'Latest question' }],
				},
			],
			tools: [],
			onTextDelta: async (delta, fullText) => {
				updates.push({ delta, fullText })
			},
		})

		expect(aiMocks.generateText).toHaveBeenCalledTimes(4)
		expect(transportMocks.obsidianFetch).toHaveBeenCalledWith(
			'https://example.com/v1/chat/completions',
			expect.objectContaining({
				method: 'POST',
				headers: expect.objectContaining({
					authorization: 'Bearer key',
				}),
			}),
		)
		expect(
			JSON.parse(transportMocks.obsidianFetch.mock.calls[0][1].body),
		).toMatchObject({
			model: 'model-1',
			messages: [
				{
					role: 'user',
					content: 'Latest question',
				},
			],
			stream: false,
		})
		expect(updates).toEqual([
			{ delta: 'Direct fallback answer', fullText: 'Direct fallback answer' },
		])
		expect(result).toMatchObject({
			message: {
				role: 'assistant',
				content: [{ type: 'text', text: 'Direct fallback answer' }],
			},
			meta: {
				usage: {
					inputTokens: 4,
					outputTokens: 3,
					totalTokens: 7,
				},
			},
		})
	})

	it('treats empty SDK assistant output as failed and uses the direct fallback', async () => {
		aiMocks.generateText
			.mockResolvedValueOnce({
				text: '',
				toolCalls: [],
				files: [],
				usage: {
					inputTokens: 2,
					outputTokens: 0,
					totalTokens: 2,
				},
				response: {},
			})
			.mockRejectedValueOnce(new Error('tools rejected'))
			.mockRejectedValueOnce(new Error('params rejected'))
			.mockRejectedValueOnce(new Error('minimal sdk rejected'))
		transportMocks.obsidianFetch.mockResolvedValueOnce(
			new Response(
				JSON.stringify({
					choices: [
						{
							message: {
								content: 'Recovered after empty SDK output',
							},
						},
					],
				}),
				{
					status: 200,
					headers: { 'content-type': 'application/json' },
				},
			),
		)

		const result = await generateAssistantTurn({
			provider: createProvider(),
			model: 'model-1',
			messages: [
				{
					role: 'user',
					content: [{ type: 'text', text: 'Hello' }],
				},
			],
			tools: [],
		})

		expect(aiMocks.generateText).toHaveBeenCalledTimes(4)
		expect(transportMocks.obsidianFetch).toHaveBeenCalledTimes(1)
		expect(result.message).toEqual({
			role: 'assistant',
			content: [{ type: 'text', text: 'Recovered after empty SDK output' }],
		})
	})

	it('passes text and reference images to image generation models', async () => {
		aiMocks.generateImage.mockResolvedValueOnce({
			image: {
				base64: Buffer.from('png-bytes').toString('base64'),
				mediaType: 'image/png',
			},
			usage: {
				inputTokens: 4,
				outputTokens: 1,
				totalTokens: 5,
			},
		})

		const result = await generateImageTurn({
			provider: createProvider(),
			model: 'model-1',
			messages: [
				{
					role: 'user',
					content: [
						{ type: 'text', text: 'Restyle this' },
						{
							type: 'image_url',
							image_url: { url: 'data:image/png;base64,AAAA' },
						},
					],
				},
			],
		})

		expect(providerMocks.createImageModel).toHaveBeenCalledTimes(1)
		expect(aiMocks.generateImage).toHaveBeenCalledWith(
			expect.objectContaining({
				prompt: {
					text: 'Restyle this',
					images: ['data:image/png;base64,AAAA'],
				},
				n: 1,
			}),
		)
		expect(result).toMatchObject({
			contentBase64: Buffer.from('png-bytes').toString('base64'),
			mediaType: 'image/png',
			prompt: 'Restyle this',
			meta: {
				providerId: 'provider-1',
				providerName: 'Provider',
				modelName: 'model-a',
				usage: {
					inputTokens: 4,
					outputTokens: 1,
					totalTokens: 5,
				},
			},
		})
	})

	it('maps image files from language model output to assistant image parts', async () => {
		aiMocks.generateText.mockResolvedValueOnce({
			text: 'Here is the image',
			toolCalls: [],
			files: [
				{
					base64: Buffer.from('png-bytes').toString('base64'),
					mediaType: 'image/png',
				},
			],
			usage: {
				inputTokens: 3,
				outputTokens: 2,
				totalTokens: 5,
			},
			response: {},
		})

		const result = await generateAssistantTurn({
			provider: createProvider(),
			model: 'model-1',
			messages: [
				{
					role: 'user',
					content: [{ type: 'text', text: 'Generate an image' }],
				},
			],
			tools: [],
		})

		expect(result.message).toEqual({
			role: 'assistant',
			content: [
				{ type: 'text', text: 'Here is the image' },
				{
					type: 'image_url',
					image_url: {
						url: `data:image/png;base64,${Buffer.from('png-bytes').toString('base64')}`,
					},
				},
			],
		})
	})

	it('uses raw response content when SDK text is empty', async () => {
		aiMocks.generateText.mockResolvedValueOnce({
			text: '',
			toolCalls: [],
			files: [],
			usage: {
				inputTokens: 3,
				outputTokens: 2,
				totalTokens: 5,
			},
			response: {
				body: {
					choices: [
						{
							message: {
								content: [
									{ type: 'text', text: 'Raw fallback answer' },
								],
							},
						},
					],
				},
			},
		})

		const result = await generateAssistantTurn({
			provider: createProvider(),
			model: 'model-1',
			messages: [
				{
					role: 'user',
					content: [{ type: 'text', text: 'Hello' }],
				},
			],
			tools: [],
		})

		expect(result.message).toEqual({
			role: 'assistant',
			content: [{ type: 'text', text: 'Raw fallback answer' }],
		})
	})
})