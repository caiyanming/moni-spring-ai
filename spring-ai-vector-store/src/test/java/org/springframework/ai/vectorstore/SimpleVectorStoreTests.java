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

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.CleanupMode;
import org.junit.jupiter.api.io.TempDir;

import org.springframework.ai.content.Media;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.Resource;
import org.springframework.util.MimeType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;
import java.time.Duration;

class SimpleVectorStoreTests {

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
		this.vectorStore = new SimpleVectorStore(SimpleVectorStore.builder(this.mockEmbeddingModel));
	}

	@Test
	void shouldAddAndRetrieveDocument() {
		Document doc = Document.builder().id("1").text("test content").metadata(Map.of("key", "value")).build();

		Mono<Void> testFlow = this.vectorStore.add(List.of(doc))
			.then(this.vectorStore.similaritySearch("test content").collectList().doOnNext(results -> {
				assertThat(results).hasSize(1).first().satisfies(result -> {
					assertThat(result.getId()).isEqualTo("1");
					assertThat(result.getText()).isEqualTo("test content");
					assertThat(result.getMetadata()).containsEntry("key", "value");
				});
			}).then());

		StepVerifier.create(testFlow).expectComplete().verify(Duration.ofSeconds(5));
	}

	@Test
	void shouldAddMultipleDocuments() {
		List<Document> docs = Arrays.asList(Document.builder().id("1").text("first").build(),
				Document.builder().id("2").text("second").build());

		Mono<Void> testFlow = this.vectorStore.add(docs)
			.then(this.vectorStore.similaritySearch("first").collectList().doOnNext(results -> {
				assertThat(results).hasSize(2).extracting(Document::getId).containsExactlyInAnyOrder("1", "2");
			}).then());

		StepVerifier.create(testFlow).expectComplete().verify(Duration.ofSeconds(5));
	}

	@Test
	void shouldHandleEmptyDocumentList() {
		StepVerifier.create(this.vectorStore.add(Collections.emptyList()))
			.expectErrorMatches(throwable -> throwable instanceof IllegalArgumentException
					&& "Documents list cannot be empty".equals(throwable.getMessage()))
			.verify(Duration.ofSeconds(5));
	}

	@Test
	void shouldHandleNullDocumentList() {
		// Test that null documents list is properly handled with meaningful error
		StepVerifier.create(this.vectorStore.add((List<Document>) null))
			.expectErrorMatches(throwable -> throwable instanceof IllegalArgumentException
					&& "Documents list cannot be null".equals(throwable.getMessage()))
			.verify(Duration.ofSeconds(5));
	}

	@Test
	void shouldDeleteDocuments() {
		Document doc = Document.builder().id("1").text("test content").build();

		Mono<Void> testFlow = this.vectorStore.add(List.of(doc))
			.then(this.vectorStore.similaritySearch("test")
				.collectList()
				.doOnNext(results -> assertThat(results).hasSize(1))
				.then())
			.then(this.vectorStore.delete(List.of("1")))
			.then(this.vectorStore.similaritySearch("test")
				.collectList()
				.doOnNext(results -> assertThat(results).isEmpty())
				.then());

		StepVerifier.create(testFlow).expectComplete().verify(Duration.ofSeconds(5));
	}

	@Test
	void shouldHandleDeleteOfNonexistentDocument() {
		Mono<Void> testFlow = this.vectorStore.delete(List.of("nonexistent-id"))
			.then(this.vectorStore.delete(List.of("nonexistent-id")));

		StepVerifier.create(testFlow).expectComplete().verify(Duration.ofSeconds(5));
	}

	@Test
	void shouldPerformSimilaritySearchWithThreshold() {
		// Configure mock to return different embeddings for different queries
		when(this.mockEmbeddingModel.embed("query")).thenReturn(Mono.just(new float[] { 0.9f, 0.9f, 0.9f }));

		Document doc = Document.builder().id("1").text("test content").build();

		SearchRequest request = SearchRequest.builder().query("query").similarityThreshold(0.99f).topK(5).build();

		Mono<Void> testFlow = this.vectorStore.add(List.of(doc))
			.then(this.vectorStore.similaritySearch(request)
				.collectList()
				.doOnNext(results -> assertThat(results).isEmpty())
				.then());

		StepVerifier.create(testFlow).expectComplete().verify(Duration.ofSeconds(5));
	}

	@Test
	void shouldSaveAndLoadVectorStore() throws IOException {
		Document doc = Document.builder()
			.id("1")
			.text("test content")
			.metadata(new HashMap<>(Map.of("key", "value")))
			.build();

		File saveFile = this.tempDir.resolve("vector-store.json").toFile();
		SimpleVectorStore loadedStore = SimpleVectorStore.builder(this.mockEmbeddingModel).build();

		Mono<Void> testFlow = this.vectorStore.add(List.of(doc))
			.doOnSuccess(v -> this.vectorStore.save(saveFile))
			.doOnSuccess(v -> loadedStore.load(saveFile))
			.then(loadedStore.similaritySearch("test content").collectList().doOnNext(results -> {
				assertThat(results).hasSize(1).first().satisfies(result -> {
					assertThat(result.getId()).isEqualTo("1");
					assertThat(result.getText()).isEqualTo("test content");
					assertThat(result.getMetadata()).containsEntry("key", "value");
				});
			}).then());

		StepVerifier.create(testFlow).expectComplete().verify(Duration.ofSeconds(5));
	}

	@Test
	void shouldHandleLoadFromInvalidResource() throws IOException {
		Resource mockResource = mock(Resource.class);
		when(mockResource.getInputStream()).thenThrow(new IOException("Resource not found"));

		assertThatThrownBy(() -> this.vectorStore.load(mockResource)).isInstanceOf(RuntimeException.class)
			.hasCauseInstanceOf(IOException.class)
			.hasMessageContaining("Resource not found");
	}

	@Test
	void shouldHandleSaveToInvalidLocation() {
		File invalidFile = new File("/invalid/path/file.json");

		assertThatThrownBy(() -> this.vectorStore.save(invalidFile)).isInstanceOf(RuntimeException.class)
			.hasCauseInstanceOf(IOException.class);
	}

	@Test
	void shouldHandleConcurrentOperations() {
		int numThreads = 10;
		List<Document> docs = Arrays.asList();
		for (int i = 0; i < numThreads; i++) {
			docs = new java.util.ArrayList<>(docs);
			docs.add(Document.builder().id(String.valueOf(i)).text("content " + i).build());
		}

		SearchRequest request = SearchRequest.builder().query("test").topK(numThreads).build();
		Set<String> expectedIds = new java.util.HashSet<>();
		for (int i = 0; i < numThreads; i++) {
			expectedIds.add(String.valueOf(i));
		}

		Mono<Void> testFlow = Flux.fromIterable(docs)
			.flatMap(doc -> this.vectorStore.add(List.of(doc)))
			.then(this.vectorStore.similaritySearch(request).collectList().doOnNext(results -> {
				assertThat(results).hasSize(numThreads);
				Set<String> resultIds = results.stream().map(Document::getId).collect(Collectors.toSet());
				assertThat(resultIds).containsExactlyInAnyOrderElementsOf(expectedIds);
				results.forEach(doc -> assertThat(doc.getText()).isEqualTo("content " + doc.getId()));
			}).then());

		StepVerifier.create(testFlow).expectComplete().verify(Duration.ofSeconds(10));
	}

	@Test
	void shouldRejectInvalidSimilarityThreshold() {
		assertThatThrownBy(() -> SearchRequest.builder().query("test").similarityThreshold(2.0f).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessage("Similarity threshold must be in [0,1] range.");
	}

	@Test
	void shouldRejectNegativeTopK() {
		assertThatThrownBy(() -> SearchRequest.builder().query("test").topK(-1).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessage("TopK should be positive.");
	}

	@Test
	void shouldHandleCosineSimilarityEdgeCases() {
		float[] zeroVector = new float[] { 0f, 0f, 0f };
		float[] normalVector = new float[] { 1f, 1f, 1f };

		assertThatThrownBy(() -> SimpleVectorStore.EmbeddingMath.cosineSimilarity(zeroVector, normalVector))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessage("Vectors cannot have zero norm");
	}

	@Test
	void shouldHandleVectorLengthMismatch() {
		float[] vector1 = new float[] { 1f, 2f };
		float[] vector2 = new float[] { 1f, 2f, 3f };

		assertThatThrownBy(() -> SimpleVectorStore.EmbeddingMath.cosineSimilarity(vector1, vector2))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessage("Vectors lengths must be equal");
	}

	@Test
	void shouldHandleNullVectors() {
		float[] vector = new float[] { 1f, 2f, 3f };

		assertThatThrownBy(() -> SimpleVectorStore.EmbeddingMath.cosineSimilarity(null, vector))
			.isInstanceOf(RuntimeException.class)
			.hasMessage("Vectors must not be null");

		assertThatThrownBy(() -> SimpleVectorStore.EmbeddingMath.cosineSimilarity(vector, null))
			.isInstanceOf(RuntimeException.class)
			.hasMessage("Vectors must not be null");
	}

	@Test
	void shouldFailNonTextDocuments() {
		Media media = new Media(MimeType.valueOf("image/png"), new ByteArrayResource(new byte[] { 0x00 }));
		Document imgDoc = Document.builder().media(media).metadata(Map.of("fileName", "pixel.png")).build();

		StepVerifier.create(this.vectorStore.add(List.of(imgDoc)))
			.expectErrorMatches(throwable -> throwable instanceof IllegalArgumentException
					&& "Only text documents are supported for now. One of the documents contains non-text content."
						.equals(throwable.getMessage()))
			.verify(Duration.ofSeconds(5));
	}

}
