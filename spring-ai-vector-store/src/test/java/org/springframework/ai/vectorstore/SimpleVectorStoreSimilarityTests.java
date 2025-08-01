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

package org.springframework.ai.vectorstore;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import org.springframework.ai.document.Document;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * @author Ilayaperumal Gopinathan
 * @author Thomas Vitale
 */
public class SimpleVectorStoreSimilarityTests {

	@Test
	public void testSimilarity() {
		Map<String, Object> metadata = new HashMap<>();
		metadata.put("foo", "bar");
		float[] testEmbedding = new float[] { 1.0f, 2.0f, 3.0f };

		SimpleVectorStoreContent storeContent = new SimpleVectorStoreContent("1", "hello, how are you?", metadata,
				testEmbedding);

		// Convert to reactive pattern using Mono
		Mono<Document> documentMono = Mono.fromCallable(() -> storeContent.toDocument(0.6)).doOnNext(document -> {
			assertThat(document).isNotNull();
			assertThat(document.getId()).isEqualTo("1");
			assertThat(document.getText()).isEqualTo("hello, how are you?");
			assertThat(document.getMetadata().get("foo")).isEqualTo("bar");
		});

		// Use StepVerifier to test the reactive chain
		StepVerifier.create(documentMono)
			.expectNextMatches(document -> document != null && "1".equals(document.getId())
					&& "hello, how are you?".equals(document.getText())
					&& "bar".equals(document.getMetadata().get("foo")))
			.expectComplete()
			.verify(Duration.ofSeconds(5));
	}

}
