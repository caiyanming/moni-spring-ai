/*
 * Copyright 2023-2025 the original author or authors.
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

package org.springframework.ai.model.tool;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import io.micrometer.observation.ObservationRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.tool.execution.DefaultToolExecutionExceptionProcessor;
import org.springframework.ai.tool.execution.ToolExecutionException;
import org.springframework.ai.tool.execution.ToolExecutionExceptionProcessor;
import org.springframework.ai.tool.observation.DefaultToolCallingObservationConvention;
import org.springframework.ai.tool.observation.ToolCallingObservationContext;
import org.springframework.ai.tool.observation.ToolCallingObservationConvention;
import org.springframework.ai.tool.observation.ToolCallingObservationDocumentation;
import org.springframework.ai.tool.resolution.DelegatingToolCallbackResolver;
import org.springframework.ai.tool.resolution.ToolCallbackResolver;
import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;

/**
 * Default implementation of {@link ToolCallingManager}.
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public final class DefaultToolCallingManager implements ToolCallingManager {

	private static final Logger logger = LoggerFactory.getLogger(DefaultToolCallingManager.class);

	// @formatter:off

	private static final ObservationRegistry DEFAULT_OBSERVATION_REGISTRY
			= ObservationRegistry.NOOP;

	private static final ToolCallingObservationConvention DEFAULT_OBSERVATION_CONVENTION
			= new DefaultToolCallingObservationConvention();

	private static final ToolCallbackResolver DEFAULT_TOOL_CALLBACK_RESOLVER
			= new DelegatingToolCallbackResolver(List.of());

	private static final ToolExecutionExceptionProcessor DEFAULT_TOOL_EXECUTION_EXCEPTION_PROCESSOR
			= DefaultToolExecutionExceptionProcessor.builder().build();

	// @formatter:on

	private final ObservationRegistry observationRegistry;

	private final ToolCallbackResolver toolCallbackResolver;

	private final ToolExecutionExceptionProcessor toolExecutionExceptionProcessor;

	private ToolCallingObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	public DefaultToolCallingManager(ObservationRegistry observationRegistry, ToolCallbackResolver toolCallbackResolver,
			ToolExecutionExceptionProcessor toolExecutionExceptionProcessor) {
		Assert.notNull(observationRegistry, "observationRegistry cannot be null");
		Assert.notNull(toolCallbackResolver, "toolCallbackResolver cannot be null");
		Assert.notNull(toolExecutionExceptionProcessor, "toolCallExceptionConverter cannot be null");

		this.observationRegistry = observationRegistry;
		this.toolCallbackResolver = toolCallbackResolver;
		this.toolExecutionExceptionProcessor = toolExecutionExceptionProcessor;
	}

	@Override
	public List<ToolDefinition> resolveToolDefinitions(ToolCallingChatOptions chatOptions) {
		Assert.notNull(chatOptions, "chatOptions cannot be null");

		List<ToolCallback> toolCallbacks = new ArrayList<>(chatOptions.getToolCallbacks());
		for (String toolName : chatOptions.getToolNames()) {
			// Skip the tool if it is already present in the request toolCallbacks.
			// That might happen if a tool is defined in the options
			// both as a ToolCallback and as a tool name.
			if (chatOptions.getToolCallbacks()
				.stream()
				.anyMatch(tool -> tool.getToolDefinition().name().equals(toolName))) {
				continue;
			}
			ToolCallback toolCallback = this.toolCallbackResolver.resolve(toolName);
			if (toolCallback == null) {
				throw new IllegalStateException("No ToolCallback found for tool name: " + toolName);
			}
			toolCallbacks.add(toolCallback);
		}

		return toolCallbacks.stream().map(ToolCallback::getToolDefinition).toList();
	}

	@Override
	public Mono<ToolExecutionResult> executeToolCalls(Prompt prompt, ChatResponse chatResponse) {
		Assert.notNull(prompt, "prompt cannot be null");
		Assert.notNull(chatResponse, "chatResponse cannot be null");

		Optional<Generation> toolCallGeneration = chatResponse.getResults()
			.stream()
			.filter(g -> !CollectionUtils.isEmpty(g.getOutput().getToolCalls()))
			.findFirst();

		if (toolCallGeneration.isEmpty()) {
			return Mono.error(new IllegalStateException("No tool call requested by the chat model"));
		}

		AssistantMessage assistantMessage = toolCallGeneration.get().getOutput();
		ToolContext toolContext = buildToolContext(prompt, assistantMessage);

		// Execute tool calls completely reactively
		return executeToolCallsReactively(prompt, assistantMessage, toolContext).map(internalResult -> {
			List<Message> conversationHistory = buildConversationHistoryAfterToolExecution(prompt.getInstructions(),
					assistantMessage, internalResult.toolResponseMessage());

			return ToolExecutionResult.builder()
				.conversationHistory(conversationHistory)
				.returnDirect(internalResult.returnDirect())
				.build();
		});
	}

	private static ToolContext buildToolContext(Prompt prompt, AssistantMessage assistantMessage) {
		Map<String, Object> toolContextMap = Map.of();

		if (prompt.getOptions() instanceof ToolCallingChatOptions toolCallingChatOptions
				&& !CollectionUtils.isEmpty(toolCallingChatOptions.getToolContext())) {
			toolContextMap = new HashMap<>(toolCallingChatOptions.getToolContext());

			toolContextMap.put(ToolContext.TOOL_CALL_HISTORY,
					buildConversationHistoryBeforeToolExecution(prompt, assistantMessage));
		}

		return new ToolContext(toolContextMap);
	}

	private static List<Message> buildConversationHistoryBeforeToolExecution(Prompt prompt,
			AssistantMessage assistantMessage) {
		List<Message> messageHistory = new ArrayList<>(prompt.copy().getInstructions());
		messageHistory.add(new AssistantMessage(assistantMessage.getText(), assistantMessage.getMetadata(),
				assistantMessage.getToolCalls()));
		return messageHistory;
	}

	/**
	 * Execute the tool calls reactively and return the response message. This method
	 * processes multiple tool calls in parallel for better performance.
	 */
	private Mono<InternalToolExecutionResult> executeToolCallsReactively(Prompt prompt,
			AssistantMessage assistantMessage, ToolContext toolContext) {
		final List<ToolCallback> toolCallbacks;
		if (prompt.getOptions() instanceof ToolCallingChatOptions toolCallingChatOptions) {
			toolCallbacks = toolCallingChatOptions.getToolCallbacks();
		}
		else {
			toolCallbacks = List.of();
		}

		final List<AssistantMessage.ToolCall> toolCalls = assistantMessage.getToolCalls();
		if (toolCalls.isEmpty()) {
			return Mono.just(new InternalToolExecutionResult(new ToolResponseMessage(List.of(), Map.of()), false));
		}

		// Execute all tool calls in parallel
		return Flux.fromIterable(toolCalls)
			.flatMap(toolCall -> executeToolCallReactively(toolCall, toolCallbacks, toolContext))
			.collectList()
			.map(toolResponses -> {
				// Determine returnDirect based on all tool callbacks
				Boolean returnDirect = null;
				for (AssistantMessage.ToolCall toolCall : toolCalls) {
					ToolCallback toolCallback = findToolCallback(toolCall.name(), toolCallbacks);
					if (toolCallback != null) {
						if (returnDirect == null) {
							returnDirect = toolCallback.getToolMetadata().returnDirect();
						}
						else {
							returnDirect = returnDirect && toolCallback.getToolMetadata().returnDirect();
						}
					}
				}
				return new InternalToolExecutionResult(new ToolResponseMessage(toolResponses, Map.of()),
						returnDirect != null ? returnDirect : false);
			});
	}

	/**
	 * Execute a single tool call reactively.
	 */
	private Mono<ToolResponseMessage.ToolResponse> executeToolCallReactively(AssistantMessage.ToolCall toolCall,
			List<ToolCallback> toolCallbacks, ToolContext toolContext) {

		logger.debug("Executing tool call reactively: {}", toolCall.name());

		String toolName = toolCall.name();
		String toolInputArguments = toolCall.arguments();

		ToolCallback toolCallback = findToolCallback(toolName, toolCallbacks);

		if (toolCallback == null) {
			return Mono.error(new IllegalStateException("No ToolCallback found for tool name: " + toolName));
		}

		ToolCallingObservationContext observationContext = ToolCallingObservationContext.builder()
			.toolDefinition(toolCallback.getToolDefinition())
			.toolMetadata(toolCallback.getToolMetadata())
			.toolCallArguments(toolInputArguments)
			.build();

		return toolCallback.call(toolInputArguments, toolContext)
			.onErrorResume(ToolExecutionException.class,
					ex -> Mono.just(this.toolExecutionExceptionProcessor.process(ex)))
			.doOnNext(observationContext::setToolCallResult)
			.map(toolCallResult -> new ToolResponseMessage.ToolResponse(toolCall.id(), toolName,
					toolCallResult != null ? toolCallResult : ""))
			.doOnNext(response -> logger.debug("Tool call completed: {} -> {}", toolName, response.responseData()));
	}

	/**
	 * Find a tool callback by name from the provided callbacks or resolver.
	 */
	private ToolCallback findToolCallback(String toolName, List<ToolCallback> toolCallbacks) {
		return toolCallbacks.stream()
			.filter(tool -> toolName.equals(tool.getToolDefinition().name()))
			.findFirst()
			.orElseGet(() -> this.toolCallbackResolver.resolve(toolName));
	}

	private List<Message> buildConversationHistoryAfterToolExecution(List<Message> previousMessages,
			AssistantMessage assistantMessage, ToolResponseMessage toolResponseMessage) {
		List<Message> messages = new ArrayList<>(previousMessages);
		messages.add(assistantMessage);
		messages.add(toolResponseMessage);
		return messages;
	}

	public void setObservationConvention(ToolCallingObservationConvention observationConvention) {
		this.observationConvention = observationConvention;
	}

	public static Builder builder() {
		return new Builder();
	}

	private record InternalToolExecutionResult(ToolResponseMessage toolResponseMessage, boolean returnDirect) {
	}

	public final static class Builder {

		private ObservationRegistry observationRegistry = DEFAULT_OBSERVATION_REGISTRY;

		private ToolCallbackResolver toolCallbackResolver = DEFAULT_TOOL_CALLBACK_RESOLVER;

		private ToolExecutionExceptionProcessor toolExecutionExceptionProcessor = DEFAULT_TOOL_EXECUTION_EXCEPTION_PROCESSOR;

		private Builder() {
		}

		public Builder observationRegistry(ObservationRegistry observationRegistry) {
			this.observationRegistry = observationRegistry;
			return this;
		}

		public Builder toolCallbackResolver(ToolCallbackResolver toolCallbackResolver) {
			this.toolCallbackResolver = toolCallbackResolver;
			return this;
		}

		public Builder toolExecutionExceptionProcessor(
				ToolExecutionExceptionProcessor toolExecutionExceptionProcessor) {
			this.toolExecutionExceptionProcessor = toolExecutionExceptionProcessor;
			return this;
		}

		public DefaultToolCallingManager build() {
			return new DefaultToolCallingManager(this.observationRegistry, this.toolCallbackResolver,
					this.toolExecutionExceptionProcessor);
		}

	}

}
