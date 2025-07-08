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

package org.springframework.ai.test.vectorstore;

import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Base test class for VectorStore implementations. Provides common test scenarios for
 * delete operations.
 *
 * @author Soby Chacko
 */
public abstract class BaseVectorStoreTests {

	/**
	 * Execute a test function with a configured VectorStore instance. This method is
	 * responsible for providing a properly initialized VectorStore within the appropriate
	 * Spring application context for testing.
	 * @param testFunction the consumer that executes test operations on the VectorStore
	 */
	protected abstract void executeTest(Consumer<VectorStore> testFunction);

	protected Document createDocument(String country, Integer year) {
		Map<String, Object> metadata = new HashMap<>();
		metadata.put("country", country);
		if (year != null) {
			metadata.put("year", year);
		}
		return new Document("The World is Big and Salvation Lurks Around the Corner", metadata);
	}

	protected Mono<List<Document>> setupTestDocuments(VectorStore vectorStore) {
		var doc1 = createDocument("BG", 2020);
		var doc2 = createDocument("NL", null);
		var doc3 = createDocument("BG", 2023);

		List<Document> documents = List.of(doc1, doc2, doc3);
		return vectorStore.add(Flux.fromIterable(documents)).then(Mono.just(documents));
	}

	private String normalizeValue(Object value) {
		return value.toString().replaceAll("^\"|\"$", "").trim();
	}

	private Mono<Void> verifyDocumentsExist(VectorStore vectorStore, List<Document> documents) {
		return vectorStore
			.similaritySearch(
					SearchRequest.builder().query("The World").topK(documents.size()).similarityThresholdAll().build())
			.collectList()
			.doOnNext(results -> assertThat(results).hasSize(documents.size()))
			.then();
	}

	private Mono<Void> verifyDocumentsDeleted(VectorStore vectorStore, List<String> deletedIds) {
		return vectorStore
			.similaritySearch(SearchRequest.builder().query("The World").topK(10).similarityThresholdAll().build())
			.map(Document::getId)
			.collectList()
			.doOnNext(foundIds -> assertThat(foundIds).doesNotContainAnyElementsOf(deletedIds))
			.then();
	}

	@Test
	protected void deleteById() {
		executeTest(vectorStore -> {
			Mono<Void> testFlow = setupTestDocuments(vectorStore)
				.flatMap(documents -> verifyDocumentsExist(vectorStore, documents).then(Mono.defer(() -> {
					List<String> idsToDelete = List.of(documents.get(0).getId(), documents.get(1).getId());
					return vectorStore.delete(Flux.fromIterable(idsToDelete))
						.then(verifyDocumentsDeleted(vectorStore, idsToDelete))
						.then(vectorStore
							.similaritySearch(
									SearchRequest.builder().query("The World").topK(5).similarityThresholdAll().build())
							.collectList()
							.doOnNext(results -> {
								assertThat(results).hasSize(1);
								assertThat(results.get(0).getId()).isEqualTo(documents.get(2).getId());
								Map<String, Object> metadata = results.get(0).getMetadata();
								assertThat(normalizeValue(metadata.get("country"))).isEqualTo("BG");
								// the values are converted into Double
								assertThat(normalizeValue(metadata.get("year"))).containsAnyOf("2023", "2023.0");
							}))
						.then(vectorStore.delete(Flux.just(documents.get(2).getId())));
				})));

			StepVerifier.create(testFlow).expectComplete().verify(Duration.ofSeconds(10));
		});
	}

	@Test
	protected void deleteWithStringFilterExpression() {
		executeTest(vectorStore -> {
			Mono<Void> testFlow = setupTestDocuments(vectorStore)
				.flatMap(documents -> verifyDocumentsExist(vectorStore, documents).then(Mono.defer(() -> {
					List<String> bgDocIds = documents.stream()
						.filter(d -> "BG".equals(d.getMetadata().get("country")))
						.map(Document::getId)
						.collect(Collectors.toList());

					return vectorStore.delete("country == 'BG'")
						.then(verifyDocumentsDeleted(vectorStore, bgDocIds))
						.then(vectorStore
							.similaritySearch(
									SearchRequest.builder().query("The World").topK(5).similarityThresholdAll().build())
							.collectList()
							.doOnNext(results -> {
								assertThat(results).hasSize(1);
								assertThat(normalizeValue(results.get(0).getMetadata().get("country"))).isEqualTo("NL");
							}))
						.then(vectorStore.delete(Flux.just(documents.get(1).getId())));
				})));

			StepVerifier.create(testFlow).expectComplete().verify(Duration.ofSeconds(10));
		});
	}

	@Test
	protected void deleteByFilter() {
		executeTest(vectorStore -> {
			Mono<Void> testFlow = setupTestDocuments(vectorStore)
				.flatMap(documents -> verifyDocumentsExist(vectorStore, documents).then(Mono.defer(() -> {
					List<String> bgDocIds = documents.stream()
						.filter(d -> "BG".equals(d.getMetadata().get("country")))
						.map(Document::getId)
						.collect(Collectors.toList());

					Filter.Expression filterExpression = new Filter.Expression(Filter.ExpressionType.EQ,
							new Filter.Key("country"), new Filter.Value("BG"));

					return vectorStore.delete(filterExpression)
						.then(verifyDocumentsDeleted(vectorStore, bgDocIds))
						.then(vectorStore
							.similaritySearch(
									SearchRequest.builder().query("The World").topK(5).similarityThresholdAll().build())
							.collectList()
							.doOnNext(results -> {
								assertThat(results).hasSize(1);
								assertThat(normalizeValue(results.get(0).getMetadata().get("country"))).isEqualTo("NL");
							}))
						.then(vectorStore.delete(Flux.just(documents.get(1).getId())));
				})));

			StepVerifier.create(testFlow).expectComplete().verify(Duration.ofSeconds(10));
		});
	}

}
