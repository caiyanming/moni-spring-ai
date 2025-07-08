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

package org.springframework.ai.chat.client.advisor.api;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Scheduler;
import reactor.core.scheduler.Schedulers;

import org.springframework.ai.chat.client.ChatClientRequest;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.AdvisorUtils;
import org.springframework.util.Assert;

/**
 * Base advisor that implements common aspects of the {@link CallAdvisor} and
 * {@link StreamAdvisor}, reducing the boilerplate code needed to implement an advisor.
 * <p>
 * It provides default implementations for the
 * {@link #adviseCall(ChatClientRequest, CallAdvisorChain)} and
 * {@link #adviseStream(ChatClientRequest, StreamAdvisorChain)} methods, delegating the
 * actual logic to the {@link #before(ChatClientRequest, AdvisorChain advisorChain)} and
 * {@link #after(ChatClientResponse, AdvisorChain advisorChain)} methods.
 *
 * @author Thomas Vitale
 * @since 1.0.0
 */
public interface BaseAdvisor extends CallAdvisor, StreamAdvisor {

	Scheduler DEFAULT_SCHEDULER = Schedulers.boundedElastic();

	@Override
	default Mono<ChatClientResponse> adviseCall(ChatClientRequest chatClientRequest,
			CallAdvisorChain callAdvisorChain) {
		Assert.notNull(chatClientRequest, "chatClientRequest cannot be null");
		Assert.notNull(callAdvisorChain, "callAdvisorChain cannot be null");

		return before(chatClientRequest, callAdvisorChain)
			.flatMap(processedRequest -> callAdvisorChain.nextCall(processedRequest))
			.flatMap(response -> after(response, callAdvisorChain))
			.subscribeOn(getScheduler());
	}

	@Override
	default Flux<ChatClientResponse> adviseStream(ChatClientRequest chatClientRequest,
			StreamAdvisorChain streamAdvisorChain) {
		Assert.notNull(chatClientRequest, "chatClientRequest cannot be null");
		Assert.notNull(streamAdvisorChain, "streamAdvisorChain cannot be null");

		return before(chatClientRequest, streamAdvisorChain)
			.flatMapMany(processedRequest -> streamAdvisorChain.nextStream(processedRequest))
			.flatMap(response -> {
				if (AdvisorUtils.onFinishReason().test(response)) {
					return after(response, streamAdvisorChain);
				}
				return Mono.just(response);
			})
			.onErrorResume(error -> Flux.error(new IllegalStateException("Stream processing failed", error)));
	}

	@Override
	default String getName() {
		return this.getClass().getSimpleName();
	}

	/**
	 * Logic to be executed before the rest of the advisor chain is called.
	 */
	Mono<ChatClientRequest> before(ChatClientRequest chatClientRequest, AdvisorChain advisorChain);

	/**
	 * Logic to be executed after the rest of the advisor chain is called.
	 */
	Mono<ChatClientResponse> after(ChatClientResponse chatClientResponse, AdvisorChain advisorChain);

	/**
	 * Scheduler used for processing the advisor logic when streaming.
	 */
	default Scheduler getScheduler() {
		return DEFAULT_SCHEDULER;
	}

}
