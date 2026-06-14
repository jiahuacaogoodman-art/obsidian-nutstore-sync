const DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1'

function trimTrailingSlashes(value: string) {
	return value.replace(/\/+$/, '')
}

function isBareOrigin(value: string) {
	try {
		const url = new URL(value)
		return url.pathname === '' || url.pathname === '/'
	} catch {
		return false
	}
}

export function getOpenAIBaseURL(api?: string) {
	const trimmed = api?.trim()
	if (!trimmed) {
		return undefined
	}
	const base = trimTrailingSlashes(trimmed)
	if (base.endsWith('/chat/completions')) {
		return base.slice(0, -'/chat/completions'.length)
	}
	return isBareOrigin(base) ? `${base}/v1` : base
}

export function getOpenAIChatCompletionURLs(api?: string) {
	const base = getOpenAIBaseURL(api) || DEFAULT_OPENAI_BASE_URL
	const candidates = [`${base}/chat/completions`]
	const trimmed = api ? trimTrailingSlashes(api.trim()) : ''
	if (trimmed.endsWith('/chat/completions')) {
		candidates.unshift(trimmed)
	} else if (trimmed && trimmed !== base) {
		candidates.push(`${trimmed}/chat/completions`)
	}
	return [...new Set(candidates)]
}