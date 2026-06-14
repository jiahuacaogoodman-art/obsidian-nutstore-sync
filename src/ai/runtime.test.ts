import { beforeEach, describe, expect, it, vi } from 'vitest'
import { generateAssistantTurn, generateImageTurn } from './runtime'
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

vi.mock('ai', () => aiMocks)

vi.mock('~/ai/providers/registry', () => ({
	getProviderResolver: () => providerMocks,
}))

vi.mock('~/ai/interleaved-message-field', () => ({
	getInterleavedMessageField: () => undefined,
}))

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
		aiMocks.generateText.mockReset()
		aiMocks.streamText.mockReset()
		aiMocks.generateImage.mockReset()
		aiMocks.stepCountIs.mockClear()
		aiMocks.tool.mockClear()
		providerMocks.createLanguageModel.mockClear()
		providerMocks.createImageModel.mockClear()
		providerMocks.assertUsable.mockClear()
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
})