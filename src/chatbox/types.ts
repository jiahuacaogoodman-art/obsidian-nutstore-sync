import type {
	ChatMessageRecord,
	ChatPendingMessage,
	ChatRunState,
	ChatTaskRecord,
	ChatToolCall,
} from '~/chat/domain'

export type {
	ChatMessageRecord,
	ChatPendingMessage,
	ChatRunState,
	ChatTaskRecord,
	ChatToolCall,
	ReversibleToolOp,
} from '~/chat/domain'

export interface ChatModelOption {
	id: string
	name: string
}

export interface ChatProviderOption {
	id: string
	name: string
	models: ChatModelOption[]
}

export interface ChatSessionHistoryItem {
	id: string
	title: string
	createdAt: number
	updatedAt: number
}

export interface ChatTimelineFragmentItem {
	id: string
	kind: 'fragment'
	createdAt: number
}

export interface ChatTimelineMessageItem {
	id: string
	kind: 'message'
	createdAt: number
	message: ChatMessageRecord
	toolCall?: ChatToolCall
}

export type ChatTimelineItem =
	| ChatTimelineFragmentItem
	| ChatTimelineMessageItem

export interface ChatImageAttachment {
	id: string
	name: string
	url: string
}

export interface ChatInferenceParams {
	temperature?: number
	maxTokens?: number
}

export interface ChatSendPayload {
	text: string
	attachments?: ChatImageAttachment[]
}

export interface ChatboxViewModel {
	title: string
	sessionHistory: ChatSessionHistoryItem[]
	activeSessionId?: string
	timeline: ChatTimelineItem[]
	currentSessionTasks: ChatTaskRecord[]
	otherSessionTasks: ChatTaskRecord[]
	providers: ChatProviderOption[]
	selectedProviderId?: string
	selectedModelId?: string
	runState: ChatRunState
	pendingMessages: ChatPendingMessage[]
	canSend: boolean
	canCreateFragment: boolean
	canCompress: boolean
	inferenceParams?: ChatInferenceParams
	selectedModelSupportsImages?: boolean
}

export interface ChatboxProps extends ChatboxViewModel {
	onNewSession: () => void
	onNewFragment: () => void
	onCompressContext: () => Promise<void>
	onSwitchSession: (sessionId: string) => void
	onDeleteSession: (sessionId: string) => Promise<void>
	onSelectProvider: (providerId: string) => void
	onSelectModel: (modelId: string) => void
	onSendMessage: (payload: ChatSendPayload) => Promise<void>
	onUpdateInferenceParams?: (params: ChatInferenceParams) => void
	onStopActiveRun?: () => void
	onCancelTask?: (taskId: string) => void
	onDeleteMessage?: (messageId: string) => void
	onRegenerateMessage?: (messageId: string) => void
	onRecallMessage?: (
		messageId: string,
		options?: { restoreFiles?: boolean },
	) => Promise<void>
	renderMarkdown?: (
		el: HTMLElement,
		markdown: string,
	) => void | (() => void) | Promise<void | (() => void)>
}

export interface ChatboxController {
	update: (props: ChatboxProps) => void
	destroy: () => void
}