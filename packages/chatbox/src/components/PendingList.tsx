import { For, Show } from 'solid-js'
import { t } from '../i18n'
import type { ChatboxProps } from '../types'

export function PendingList(props: {
	pendingMessages: ChatboxProps['pendingMessages']
}) {
	return (
		<Show when={props.pendingMessages.length > 0}>
			<div class="rounded-3 border border-dashed border-[var(--background-modifier-border)] bg-[var(--background-primary-alt)] p-3">
				<div class="text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
					{t('pendingMessages')}
				</div>
				<div class="mt-2 flex flex-col gap-2">
					<For each={props.pendingMessages}>
						{(message) => (
							<div class="rounded-2 bg-[var(--background-secondary)] p-3 text-sm text-[var(--text-normal)] whitespace-pre-wrap break-words select-text">
								<Show when={message.text}>{message.text}</Show>
								<Show when={message.attachments?.length}>
									<div class="mt-2 flex gap-2 overflow-x-auto pb-1 scrollbar-default">
										<For each={message.attachments || []}>
											{(attachment) => (
												<img
													class="h-14 w-14 shrink-0 rounded-2 border border-[var(--background-modifier-border)] object-cover"
													src={attachment.url}
													alt={attachment.name}
												/>
											)}
										</For>
									</div>
								</Show>
							</div>
						)}
					</For>
				</div>
			</div>
		</Show>
	)
}