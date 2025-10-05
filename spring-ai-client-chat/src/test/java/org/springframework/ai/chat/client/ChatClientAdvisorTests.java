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

package org.springframework.ai.chat.client;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Captor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import org.springframework.ai.chat.client.advisor.PromptChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.BDDMockito.given;

/**
 * Tests for the ChatClient with a focus on verifying the handling of conversation memory
 * and the integration of PromptChatMemoryAdvisor to ensure accurate responses based on
 * previous interactions.
 *
 * @author Christian Tzolov
 * @author Alexandros Pappas
 */
@ExtendWith(MockitoExtension.class)
public class ChatClientAdvisorTests {

	@Mock
	ChatModel chatModel;

	@Captor
	ArgumentCaptor<Prompt> promptCaptor;

	private String join(Flux<String> fluxContent) {
		AtomicReference<String> joined = new AtomicReference<>("");
		StepVerifier.create(fluxContent.collectList())
			.consumeNextWith(values -> joined.set(values.stream().collect(Collectors.joining())))
			.verifyComplete();
		return joined.get();
	}

	private void awaitMemoryEntry(ChatMemory chatMemory, MessageType messageType, String expectedText) {
		StepVerifier.create(Mono.defer(() -> {
			List<Message> messages = chatMemory.get(ChatMemory.DEFAULT_CONVERSATION_ID);
			boolean present = messages.stream()
				.filter(message -> message.getMessageType() == messageType)
				.anyMatch(message -> message.getText() != null && message.getText().contains(expectedText));
			return present ? Mono.just(true) : Mono.empty();
		}).repeatWhenEmpty(repeat -> repeat.delayElements(Duration.ofMillis(10)).take(50)))
			.expectNext(true)
			.verifyComplete();
	}

	@Test
	public void promptChatMemory() {

		// Create a ChatResponseMetadata instance with default values
		ChatResponseMetadata chatResponseMetadata = ChatResponseMetadata.builder().build();

		// Mock the chatModel to return predefined ChatResponse objects when called
		given(this.chatModel.call(this.promptCaptor.capture()))
			.willReturn(Mono.just(new ChatResponse(List.of(new Generation(new AssistantMessage("Hello John"))),
					chatResponseMetadata)))
			.willReturn(Mono.just(new ChatResponse(List.of(new Generation(new AssistantMessage("Your name is John"))),
					chatResponseMetadata)));

		// Initialize a message window chat memory to store conversation history
		ChatMemory chatMemory = MessageWindowChatMemory.builder()
			.chatMemoryRepository(new InMemoryChatMemoryRepository())
			.build();

		// Build a ChatClient with default system text and a memory advisor
		var chatClient = ChatClient.builder(this.chatModel)
			.defaultSystem("Default system text.")
			.defaultAdvisors(PromptChatMemoryAdvisor.builder(chatMemory).build())
			.build();

		// Simulate a user prompt and verify the response
		StepVerifier.create(chatClient.prompt().user("my name is John").call().chatResponse())
			.assertNext(
					chatResponse -> assertThat(chatResponse.getResult().getOutput().getText()).isEqualTo("Hello John"))
			.verifyComplete();

		// Capture and verify the system message instructions
		Message systemMessage = this.promptCaptor.getValue().getInstructions().get(0);
		assertThat(systemMessage.getText()).isEqualToIgnoringWhitespace("""
				Default system text.

				Use the conversation memory from the MEMORY section to provide accurate answers.

				---------------------
				MEMORY:
				---------------------
				""");
		assertThat(systemMessage.getMessageType()).isEqualTo(MessageType.SYSTEM);

		// Capture and verify the user message instructions
		Message userMessage = this.promptCaptor.getValue().getInstructions().get(1);
		assertThat(userMessage.getText()).isEqualToIgnoringWhitespace("my name is John");

		// Simulate another user prompt and verify the response
		StepVerifier.create(chatClient.prompt().user("What is my name?").call().content())
			.expectNext("Your name is John")
			.verifyComplete();

		// Capture and verify the updated system message instructions
		systemMessage = this.promptCaptor.getValue().getInstructions().get(0);
		assertThat(systemMessage.getText()).isEqualToIgnoringWhitespace("""
				Default system text.

				Use the conversation memory from the MEMORY section to provide accurate answers.

				---------------------
				MEMORY:
				USER:my name is John
				ASSISTANT:Hello John
				---------------------
				""");
		assertThat(systemMessage.getMessageType()).isEqualTo(MessageType.SYSTEM);

		// Capture and verify the updated user message instructions
		userMessage = this.promptCaptor.getValue().getInstructions().get(1);
		assertThat(userMessage.getText()).isEqualToIgnoringWhitespace("What is my name?");
	}

	@Test
	public void streamingPromptChatMemory() {

		// Mock the chatModel to stream predefined ChatResponse objects
		given(this.chatModel.stream(this.promptCaptor.capture())).willReturn(Flux.generate(
				() -> new ChatResponse(List.of(new Generation(new AssistantMessage("Hello John")))), (state, sink) -> {
					sink.next(state);
					sink.complete();
					return state;
				}))
			.willReturn(Flux.generate(
					() -> new ChatResponse(List.of(new Generation(new AssistantMessage("Your name is John")))),
					(state, sink) -> {
						sink.next(state);
						sink.complete();
						return state;
					}));

		// Initialize a message window chat memory to store conversation history
		ChatMemory chatMemory = MessageWindowChatMemory.builder()
			.chatMemoryRepository(new InMemoryChatMemoryRepository())
			.build();

		// Build a ChatClient with default system text and a memory advisor
		var chatClient = ChatClient.builder(this.chatModel)
			.defaultSystem("Default system text.")
			.defaultAdvisors(PromptChatMemoryAdvisor.builder(chatMemory).build())
			.build();

		// Simulate a streaming user prompt and verify the response
		var content = join(chatClient.prompt().user("my name is John").stream().content());

		// Assert that the streamed content matches the expected output
		assertThat(content).isEqualTo("Hello John");

		// Capture and verify the system message instructions
		Message systemMessage = this.promptCaptor.getValue().getInstructions().get(0);
		assertThat(systemMessage.getText()).isEqualToIgnoringWhitespace("""
				Default system text.

				Use the conversation memory from the MEMORY section to provide accurate answers.

				---------------------
				MEMORY:
				---------------------
				""");
		assertThat(systemMessage.getMessageType()).isEqualTo(MessageType.SYSTEM);

		// Capture and verify the user message instructions
		Message userMessage = this.promptCaptor.getValue().getInstructions().get(1);
		assertThat(userMessage.getText()).isEqualToIgnoringWhitespace("my name is John");

		awaitMemoryEntry(chatMemory, MessageType.ASSISTANT, "Hello John");

		// Simulate another streaming user prompt and verify the response
		content = join(chatClient.prompt().user("What is my name?").stream().content());

		// Assert that the streamed content matches the expected output
		assertThat(content).isEqualTo("Your name is John");

		// Capture and verify the updated system message instructions
		systemMessage = this.promptCaptor.getValue().getInstructions().get(0);
		assertThat(systemMessage.getText()).isEqualToIgnoringWhitespace("""
				Default system text.

				Use the conversation memory from the MEMORY section to provide accurate answers.

				---------------------
				MEMORY:
				USER:my name is John
				ASSISTANT:Hello John
				---------------------
				""");
		assertThat(systemMessage.getMessageType()).isEqualTo(MessageType.SYSTEM);

		// Capture and verify the updated user message instructions
		userMessage = this.promptCaptor.getValue().getInstructions().get(1);
		assertThat(userMessage.getText()).isEqualToIgnoringWhitespace("What is my name?");
	}

}
