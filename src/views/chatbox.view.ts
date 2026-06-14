import { mount as mountChatbox } from 'chatbox'
import { Component, ItemView, MarkdownRenderer, WorkspaceLeaf } from 'obsidian'
import type { ChatboxController, ChatboxProps } from '~/chatbox/types'
import i18n from '~/i18n'
import logger from '~/utils/logger'
import type NutstorePlugin from '..'

export const CHATBOX_VIEW_TYPE = 'guozha-ai-pro-chatbox'

export default class ChatboxView extends ItemView {
	private rootEl!: HTMLDivElement
	private controller?: ChatboxController
	private unsub?: () => void
	private readonly renderMarkdown: NonNullable<ChatboxProps['renderMarkdown']> =
		async (el: HTMLElement, markdown: string) => {
			const component = new Component()
			this.addChild(component)
			component.load()

			const fallbackText = markdown
			const renderedEl = document.createElement('div')

			try {
				await MarkdownRenderer.render(
					this.app,
					markdown,
					renderedEl,
					'',
					component,
				)
			} catch (error) {
				logger.error('Error rendering chat markdown:', error)
				component.unload()
				el.replaceChildren()
				el.textContent = fallbackText
				return
			}

			el.replaceChildren(...Array.from(renderedEl.childNodes))
			if (!el.childNodes.length) {
				el.textContent = fallbackText
			}

			return () => {
				component.unload()
				el.replaceChildren()
			}
		}

	constructor(
		leaf: WorkspaceLeaf,
		private plugin: NutstorePlugin,
	) {
		super(leaf)
	}

	getViewType() {
		return CHATBOX_VIEW_TYPE
	}

	getDisplayText() {
		return i18n.t('chatbox.title')
	}

	getIcon() {
		return 'bot'
	}

	private getChatboxProps(): ChatboxProps {
		return {
			...this.plugin.chatService.getViewProps(),
			renderMarkdown: this.renderMarkdown,
		}
	}

	async onOpen() {
		this.contentEl.empty()
		this.rootEl = this.contentEl.createDiv({
			cls: 'guozha-chatbox-view h-full',
		})
		await this.plugin.chatService.ensureSession()
		this.controller = mountChatbox(this.rootEl, this.getChatboxProps())
		this.unsub = this.plugin.chatService.subscribe(() => {
			this.controller?.update(this.getChatboxProps())
		})
	}

	async onClose() {
		this.unsub?.()
		this.controller?.destroy()
		this.contentEl.empty()
	}
}