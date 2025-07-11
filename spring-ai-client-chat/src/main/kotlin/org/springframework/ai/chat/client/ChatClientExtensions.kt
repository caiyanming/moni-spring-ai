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

package org.springframework.ai.chat.client

import org.springframework.ai.chat.model.ChatResponse
import org.springframework.core.ParameterizedTypeReference
import reactor.core.publisher.Mono

/**
 * Extensions for [ChatClient] providing a reified generic adapters for `entity` and `responseEntity`
 *
 * @author Josh Long
 */

inline fun <reified T> ChatClient.CallResponseSpec.entity(): Mono<T> =
	entity(object : ParameterizedTypeReference<T>() {})

inline fun <reified T> ChatClient.CallResponseSpec.responseEntity(): Mono<ResponseEntity<ChatResponse, T>> =
	responseEntity(object : ParameterizedTypeReference<T>() {}) 
