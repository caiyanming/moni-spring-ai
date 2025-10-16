package org.springframework.ai.chat.client;

/**
 * Shared Reactor context keys used by {@link ChatClient} implementations.
 */
public final class ChatClientContextKeys {

	private ChatClientContextKeys() {
	}

	/**
	 * Reactor context key that stores the {@link ChatClientResponse} produced by a chat
	 * call.
	 */
	public static final String CHAT_CLIENT_RESPONSE = "spring.ai.chatClientResponse";

}
