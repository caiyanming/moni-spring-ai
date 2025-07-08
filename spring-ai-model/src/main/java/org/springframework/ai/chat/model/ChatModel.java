/*
 * Copyright 2023-2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.chat.model;

import java.util.Arrays;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.Model;

/**
 * The ChatModel interface provides a reactive API for chat-based AI model interactions.
 * It extends the base {@link Model} interface with chat-specific functionality and
 * streaming capabilities.
 *
 * <p>
 * This interface is fully reactive, returning {@link Mono} or {@link Flux} for
 * non-blocking operations.
 *
 * @author Mark Pollack
 * @author Christian Tzolov
 * @since 0.8.0
 */
public interface ChatModel extends Model<Prompt, ChatResponse>, StreamingChatModel {

	/**
	 * Executes a reactive chat call with a simple string message.
	 * @param message the user message string
	 * @return a {@link Mono} containing the response text
	 */
	default Mono<String> call(String message) {
		Prompt prompt = new Prompt(new UserMessage(message));
		return call(prompt).map(response -> {
			Generation generation = response.getResult();
			return (generation != null) ? generation.getOutput().getText() : "";
		});
	}

	/**
	 * Executes a reactive chat call with multiple messages.
	 * @param messages the array of messages
	 * @return a {@link Mono} containing the response text
	 */
	default Mono<String> call(Message... messages) {
		Prompt prompt = new Prompt(Arrays.asList(messages));
		return call(prompt).map(response -> {
			Generation generation = response.getResult();
			return (generation != null) ? generation.getOutput().getText() : "";
		});
	}

	/**
	 * Executes a reactive chat call with a {@link Prompt}.
	 * @param prompt the chat prompt
	 * @return a {@link Mono} containing the {@link ChatResponse}
	 */
	@Override
	Mono<ChatResponse> call(Prompt prompt);

	default ChatOptions getDefaultOptions() {
		return ChatOptions.builder().build();
	}

	default Flux<ChatResponse> stream(Prompt prompt) {
		throw new UnsupportedOperationException("streaming is not supported");
	}

}
