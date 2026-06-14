import { describe, expect, it } from 'vitest'
import { getOpenAIBaseURL, getOpenAIChatCompletionURLs } from './openai-base-url'

describe('OpenAI base URL helpers', () => {
	it('adds /v1 for bare OpenAI-compatible origins', () => {
		expect(getOpenAIBaseURL('https://example.com')).toBe(
			'https://example.com/v1',
		)
		expect(getOpenAIChatCompletionURLs('https://example.com')).toEqual([
			'https://example.com/v1/chat/completions',
			'https://example.com/chat/completions',
		])
	})

	it('keeps explicit API path values intact', () => {
		expect(getOpenAIBaseURL('https://example.com/v1')).toBe(
			'https://example.com/v1',
		)
		expect(getOpenAIChatCompletionURLs('https://example.com/v1')).toEqual([
			'https://example.com/v1/chat/completions',
		])
	})
})