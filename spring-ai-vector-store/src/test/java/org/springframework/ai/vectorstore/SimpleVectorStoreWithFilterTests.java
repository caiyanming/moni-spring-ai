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

package org.springframework.ai.vectorstore;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.CleanupMode;
import org.junit.jupiter.api.io.TempDir;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.filter.Filter;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.AND;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.EQ;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.GTE;
import static org.springframework.ai.vectorstore.filter.Filter.ExpressionType.LTE;

/**
 * @author Jemin Huh
 */
class SimpleVectorStoreWithFilterTests {

	@TempDir(cleanup = CleanupMode.ON_SUCCESS)
	Path tempDir;

	private SimpleVectorStore vectorStore;

	private EmbeddingModel mockEmbeddingModel;

	@BeforeEach
	void setUp() {
		this.mockEmbeddingModel = mock(EmbeddingModel.class);
		when(this.mockEmbeddingModel.dimensions()).thenReturn(Mono.just(3));
		when(this.mockEmbeddingModel.embed(any(String.class))).thenReturn(Mono.just(new float[] { 0.1f, 0.2f, 0.3f }));
		when(this.mockEmbeddingModel.embed(any(Document.class)))
			.thenReturn(Mono.just(new float[] { 0.1f, 0.2f, 0.3f }));
		this.vectorStore = SimpleVectorStore.builder(this.mockEmbeddingModel).build();
	}

	@Test
	void shouldAddAndRetrieveDocumentWithFilter() {
		Document doc = Document.builder()
			.id("1")
			.text("test content")
			.metadata(Map.of("country", "BG", "year", 2020, "activationDate", "1970-01-01T00:00:02Z"))
			.build();

		StepVerifier.create(
				this.vectorStore.add(List.of(doc))
					.then(this.vectorStore.similaritySearch(
							SearchRequest.builder().query("test content").filterExpression("country == 'BG'").build())
						.collectList())
					.doOnNext(results -> {
						assertThat(results).hasSize(1).first().satisfies(result -> {
							assertThat(result.getId()).isEqualTo("1");
							assertThat(result.getText()).isEqualTo("test content");
							assertThat(result.getMetadata()).hasSize(4);
							assertThat(result.getMetadata()).containsEntry("distance", 2.220446049250313E-16);
						});
					})
					.then(this.vectorStore.similaritySearch(
							SearchRequest.builder().query("test content").filterExpression("country == 'KR'").build())
						.collectList())
					.doOnNext(results -> {
						assertThat(results).hasSize(0);
					})
					.then(this.vectorStore
						.similaritySearch(SearchRequest.builder()
							.query("test content")
							.filterExpression("country == 'BG' && year == 2020")
							.build())
						.collectList())
					.doOnNext(results -> {
						assertThat(results).hasSize(1).first().satisfies(result -> {
							assertThat(result.getId()).isEqualTo("1");
							assertThat(result.getText()).isEqualTo("test content");
							assertThat(result.getMetadata()).hasSize(4);
							assertThat(result.getMetadata()).containsEntry("distance", 2.220446049250313E-16);
						});
					})
					.then(this.vectorStore
						.similaritySearch(SearchRequest.builder()
							.query("test content")
							.filterExpression("country == 'BG' && year == 2024")
							.build())
						.collectList())
					.doOnNext(results -> {
						assertThat(results).hasSize(0);
					})
					.then(this.vectorStore
						.similaritySearch(SearchRequest.builder()
							.query("test content")
							.filterExpression("country in ['BG', 'NL']")
							.build())
						.collectList())
					.doOnNext(results -> {
						assertThat(results).hasSize(1).first().satisfies(result -> {
							assertThat(result.getId()).isEqualTo("1");
							assertThat(result.getText()).isEqualTo("test content");
							assertThat(result.getMetadata()).hasSize(4);
							assertThat(result.getMetadata()).containsEntry("distance", 2.220446049250313E-16);
						});
					})
					.then(this.vectorStore
						.similaritySearch(SearchRequest.builder()
							.query("test content")
							.filterExpression("country in ['KR', 'NL']")
							.build())
						.collectList())
					.doOnNext(results -> {
						assertThat(results).hasSize(0);
					})
					.then(this.vectorStore
						.similaritySearch(SearchRequest.builder()
							.query("test content")
							.filterExpression(new Filter.Expression(EQ, new Filter.Key("activationDate"),
									new Filter.Value(new Date(2000))))
							.build())
						.collectList())
					.doOnNext(results -> {
						assertThat(results).hasSize(1).first().satisfies(result -> {
							assertThat(result.getId()).isEqualTo("1");
							assertThat(result.getText()).isEqualTo("test content");
							assertThat(result.getMetadata()).hasSize(4);
							assertThat(result.getMetadata()).containsEntry("distance", 2.220446049250313E-16);
						});
					})
					.then(this.vectorStore
						.similaritySearch(SearchRequest.builder()
							.query("test content")
							.filterExpression(new Filter.Expression(EQ, new Filter.Key("activationDate"),
									new Filter.Value(new Date(3000))))
							.build())
						.collectList())
					.doOnNext(results -> {
						assertThat(results).hasSize(0);
					}))
			.expectComplete()
			.verify(Duration.ofSeconds(5));
	}

	@Test
	void shouldAddMultipleDocumentsWithFilter() {
		List<Document> docs = Arrays.asList(
				Document.builder()
					.id("1")
					.text("first")
					.metadata(Map.of("country", "BG", "year", 2020, "activationDate", "1970-01-01T00:00:02Z"))
					.build(),
				Document.builder()
					.id("2")
					.text("second")
					.metadata(Map.of("country", "KR", "year", 2022, "activationDate", "1970-01-01T00:00:03Z"))
					.build());

		StepVerifier.create(this.vectorStore.add(docs)
			.then(this.vectorStore.similaritySearch("first").collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(2).extracting(Document::getId).containsExactlyInAnyOrder("1", "2");
			})
			.then(this.vectorStore
				.similaritySearch(SearchRequest.builder().query("first").filterExpression("country == 'BG'").build())
				.collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(1).first().satisfies(result -> {
					assertThat(result.getId()).isEqualTo("1");
					assertThat(result.getText()).isEqualTo("first");
					assertThat(result.getMetadata()).hasSize(4);
					assertThat(result.getMetadata()).containsEntry("distance", 2.220446049250313E-16);
				});
			})
			.then(this.vectorStore
				.similaritySearch(SearchRequest.builder().query("first").filterExpression("country == 'NL'").build())
				.collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(0);
			})
			.then(this.vectorStore.similaritySearch(
					SearchRequest.builder().query("first").filterExpression("country == 'BG' && year == 2020").build())
				.collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(1).first().satisfies(result -> {
					assertThat(result.getId()).isEqualTo("1");
					assertThat(result.getText()).isEqualTo("first");
					assertThat(result.getMetadata()).hasSize(4);
					assertThat(result.getMetadata()).containsEntry("distance", 2.220446049250313E-16);
				});
			})
			.then(this.vectorStore.similaritySearch(
					SearchRequest.builder().query("first").filterExpression("country == 'KR' && year == 2022").build())
				.collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(1).first().satisfies(result -> {
					assertThat(result.getId()).isEqualTo("2");
					assertThat(result.getText()).isEqualTo("second");
					assertThat(result.getMetadata()).hasSize(4);
					assertThat(result.getMetadata()).containsEntry("distance", 2.220446049250313E-16);
				});
			})
			.then(this.vectorStore
				.similaritySearch(SearchRequest.builder()
					.query("test content")
					.filterExpression("country == 'KR' && year == 2024")
					.build())
				.collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(0);
			})
			.then(this.vectorStore
				.similaritySearch(
						SearchRequest.builder().query("first").filterExpression("country in ['BG', 'NL']").build())
				.collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(1).first().satisfies(result -> {
					assertThat(result.getId()).isEqualTo("1");
					assertThat(result.getText()).isEqualTo("first");
					assertThat(result.getMetadata()).hasSize(4);
					assertThat(result.getMetadata()).containsEntry("distance", 2.220446049250313E-16);
				});
			})
			.then(this.vectorStore
				.similaritySearch(
						SearchRequest.builder().query("first").filterExpression("country in ['KR', 'NL']").build())
				.collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(1);
			})
			.then(this.vectorStore
				.similaritySearch(SearchRequest.builder()
					.query("first")
					.filterExpression(new Filter.Expression(EQ, new Filter.Key("activationDate"),
							new Filter.Value(new Date(2000))))
					.build())
				.collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(1).first().satisfies(result -> {
					assertThat(result.getId()).isEqualTo("1");
					assertThat(result.getText()).isEqualTo("first");
					assertThat(result.getMetadata()).hasSize(4);
					assertThat(result.getMetadata()).containsEntry("distance", 2.220446049250313E-16);
				});
			})
			.then(this.vectorStore.similaritySearch(SearchRequest.builder()
				.query("first")
				.filterExpression(new Filter.Expression(AND,
						new Filter.Expression(GTE, new Filter.Key("activationDate"), new Filter.Value(new Date(2000))),
						new Filter.Expression(LTE, new Filter.Key("activationDate"), new Filter.Value(new Date(3000)))))
				.build()).collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(2).first().satisfies(result -> {
					assertThat(result.getId()).isEqualTo("1");
					assertThat(result.getText()).isEqualTo("first");
					assertThat(result.getMetadata()).hasSize(4);
					assertThat(result.getMetadata()).containsEntry("distance", 2.220446049250313E-16);
				});
			})
			.then(this.vectorStore
				.similaritySearch(SearchRequest.builder()
					.query("test content")
					.filterExpression(new Filter.Expression(EQ, new Filter.Key("activationDate"),
							new Filter.Value(new Date(3000))))
					.build())
				.collectList())
			.doOnNext(results -> {
				assertThat(results).hasSize(1);
			})).expectComplete().verify(Duration.ofSeconds(5));
	}

}
