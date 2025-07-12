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

package org.springframework.ai.vectorstore.qdrant;

import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
import io.qdrant.client.grpc.Collections.Distance;
import io.qdrant.client.grpc.Collections.VectorParams;
import io.qdrant.client.grpc.Collections.VectorsConfig;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;
import org.testcontainers.qdrant.QdrantContainer;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import org.springframework.ai.content.Media;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.DocumentMetadata;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.openai.OpenAiEmbeddingModel;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.test.vectorstore.BaseVectorStoreTests;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Reactive integration tests for QdrantVectorStore.
 * <p>
 * All tests are converted to use reactive patterns with StepVerifier instead of blocking
 * calls to ensure proper reactive behavior.
 *
 * @author Anush Shetty
 * @author Josh Long
 * @author Eddú Meléndez
 * @author Thomas Vitale
 * @author Soby Chacko
 * @author Jonghoon Park
 * @author Kim San
 * @author MoniAI Team
 * @since 0.8.1
 */
@Testcontainers
@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
public class QdrantVectorStoreIT extends BaseVectorStoreTests {

	private static final String COLLECTION_NAME = "test_collection";

	private static final int EMBEDDING_DIMENSION = 1536;

	private static final Duration TIMEOUT = Duration.ofSeconds(30);

	@Container
	static QdrantContainer qdrantContainer = new QdrantContainer(QdrantImage.DEFAULT_IMAGE);

	private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
		.withUserConfiguration(TestApplication.class);

	List<Document> documents = List.of(
			new Document("Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!!",
					Collections.singletonMap("meta1", "meta1")),
			new Document("Hello World Hello World Hello World Hello World Hello World Hello World Hello World",
					Collections.singletonMap("meta1", List.of("meta-list"))),
			new Document(
					"Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression",
					Collections.singletonMap("meta2", List.of("meta-list"))));

	@BeforeAll
	static void setup() throws InterruptedException, ExecutionException {

		String host = qdrantContainer.getHost();
		int port = qdrantContainer.getGrpcPort();
		QdrantClient client = new QdrantClient(QdrantGrpcClient.newBuilder(host, port, false).build());

		// Use VectorsConfig instead of direct VectorParams
		var vectorParams = VectorParams.newBuilder().setDistance(Distance.Cosine).setSize(EMBEDDING_DIMENSION).build();
		var vectorsConfig = VectorsConfig.newBuilder().setParams(vectorParams).build();

		client.createCollectionAsync(COLLECTION_NAME, vectorsConfig).get();

		client.close();
	}

	@Override
	protected void executeTest(Consumer<VectorStore> testFunction) {
		this.contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);
			testFunction.accept(vectorStore);
		});
	}

	@Test
	public void addAndSearch() {
		this.contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);

			// Reactive: Add documents
			StepVerifier.create(vectorStore.add(this.documents)).expectComplete().verify(TIMEOUT);

			// Reactive: Search documents
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.builder().query("Great").topK(1).build())
					.collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					Document resultDoc = results.get(0);
					assertThat(resultDoc.getId()).isEqualTo(this.documents.get(2).getId());
					assertThat(resultDoc.getText()).isEqualTo(
							"Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression");
					assertThat(resultDoc.getMetadata()).containsKeys("meta2", DocumentMetadata.DISTANCE.value());
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Delete documents
			StepVerifier.create(vectorStore.delete(this.documents.stream().map(Document::getId).toList()))
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Verify deletion
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.builder().query("Great").topK(1).build())
					.collectList())
				.assertNext(results -> assertThat(results).hasSize(0))
				.expectComplete()
				.verify(TIMEOUT);
		});
	}

	@Test
	public void addAndSearchWithFilters() {
		this.contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);

			var bgDocument = new Document("The World is Big and Salvation Lurks Around the Corner",
					Map.of("country", "Bulgaria", "number", 3));
			var nlDocument = new Document("The World is Big and Salvation Lurks Around the Corner",
					Map.of("country", "Netherlands", "number", 90));

			// Reactive: Add documents
			StepVerifier.create(vectorStore.add(List.of(bgDocument, nlDocument))).expectComplete().verify(TIMEOUT);

			var request = SearchRequest.builder().query("The World").topK(5).build();

			// Reactive: Search without filters
			StepVerifier.create(vectorStore.similaritySearch(request).collectList())
				.assertNext(results -> assertThat(results).hasSize(2))
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Search with Bulgaria filter
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.from(request)
					.similarityThresholdAll()
					.filterExpression("country == 'Bulgaria'")
					.build()).collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					assertThat(results.get(0).getId()).isEqualTo(bgDocument.getId());
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Search with Netherlands filter
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.from(request)
					.similarityThresholdAll()
					.filterExpression("country == 'Netherlands'")
					.build()).collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					assertThat(results.get(0).getId()).isEqualTo(nlDocument.getId());
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Search with NOT filter
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.from(request)
					.similarityThresholdAll()
					.filterExpression("NOT(country == 'Netherlands')")
					.build()).collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					assertThat(results.get(0).getId()).isEqualTo(bgDocument.getId());
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Search with IN filter
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.from(request)
					.similarityThresholdAll()
					.filterExpression("number in [3, 5, 12]")
					.build()).collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					assertThat(results.get(0).getId()).isEqualTo(bgDocument.getId());
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Search with NIN filter
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.from(request)
					.similarityThresholdAll()
					.filterExpression("number nin [3, 5, 12]")
					.build()).collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					assertThat(results.get(0).getId()).isEqualTo(nlDocument.getId());
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Clean up
			StepVerifier
				.create(vectorStore.delete(List.of(bgDocument, nlDocument).stream().map(Document::getId).toList()))
				.expectComplete()
				.verify(TIMEOUT);
		});
	}

	@Test
	public void documentUpdateTest() {
		this.contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);

			Document document = new Document(UUID.randomUUID().toString(), "Spring AI rocks!!",
					Collections.singletonMap("meta1", "meta1"));

			// Reactive: Add document
			StepVerifier.create(vectorStore.add(document)).expectComplete().verify(TIMEOUT);

			// Reactive: Search for original document
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.builder().query("Spring").topK(5).build())
					.collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					Document resultDoc = results.get(0);
					assertThat(resultDoc.getId()).isEqualTo(document.getId());
					assertThat(resultDoc.getText()).isEqualTo("Spring AI rocks!!");
					assertThat(resultDoc.getMetadata()).containsKey("meta1");
					assertThat(resultDoc.getMetadata()).containsKey(DocumentMetadata.DISTANCE.value());
				})
				.expectComplete()
				.verify(TIMEOUT);

			Document sameIdDocument = new Document(document.getId(),
					"The World is Big and Salvation Lurks Around the Corner",
					Collections.singletonMap("meta2", "meta2"));

			// Reactive: Update document (same ID)
			StepVerifier.create(vectorStore.add(sameIdDocument)).expectComplete().verify(TIMEOUT);

			// Reactive: Search for updated document
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.builder().query("FooBar").topK(5).build())
					.collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					Document resultDoc = results.get(0);
					assertThat(resultDoc.getId()).isEqualTo(document.getId());
					assertThat(resultDoc.getText()).isEqualTo("The World is Big and Salvation Lurks Around the Corner");
					assertThat(resultDoc.getMetadata()).containsKey("meta2");
					assertThat(resultDoc.getMetadata()).containsKey(DocumentMetadata.DISTANCE.value());
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Clean up
			StepVerifier.create(vectorStore.delete(List.of(document.getId()))).expectComplete().verify(TIMEOUT);
		});
	}

	@Test
	public void searchThresholdTest() {
		this.contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);

			// Reactive: Add documents
			StepVerifier.create(vectorStore.add(this.documents)).expectComplete().verify(TIMEOUT);

			var request = SearchRequest.builder().query("Great").topK(5).build();

			// Reactive: Get full results to calculate threshold
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.from(request).similarityThresholdAll().build())
					.collectList())
				.assertNext(fullResult -> {
					List<Double> scores = fullResult.stream().map(Document::getScore).toList();
					assertThat(scores).hasSize(3);

					double similarityThreshold = (scores.get(0) + scores.get(1)) / 2;

					// Reactive: Search with threshold
					StepVerifier
						.create(vectorStore
							.similaritySearch(
									SearchRequest.from(request).similarityThreshold(similarityThreshold).build())
							.collectList())
						.assertNext(results -> {
							assertThat(results).hasSize(1);
							Document resultDoc = results.get(0);
							assertThat(resultDoc.getId()).isEqualTo(this.documents.get(2).getId());
							assertThat(resultDoc.getText()).isEqualTo(
									"Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression");
							assertThat(resultDoc.getMetadata()).containsKey("meta2");
							assertThat(resultDoc.getMetadata()).containsKey(DocumentMetadata.DISTANCE.value());
							assertThat(resultDoc.getScore()).isGreaterThanOrEqualTo(similarityThreshold);
						})
						.expectComplete()
						.verify(TIMEOUT);
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Clean up
			StepVerifier.create(vectorStore.delete(this.documents.stream().map(Document::getId).toList()))
				.expectComplete()
				.verify(TIMEOUT);
		});
	}

	@Test
	void deleteWithComplexFilterExpression() {
		this.contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);

			var doc1 = new Document("Content 1", Map.of("type", "A", "priority", 1));
			var doc2 = new Document("Content 2", Map.of("type", "A", "priority", 2));
			var doc3 = new Document("Content 3", Map.of("type", "B", "priority", 1));

			// Reactive: Add documents
			StepVerifier.create(vectorStore.add(List.of(doc1, doc2, doc3))).expectComplete().verify(TIMEOUT);

			// Complex filter expression: (type == 'A' AND priority > 1)
			Filter.Expression priorityFilter = new Filter.Expression(Filter.ExpressionType.GT,
					new Filter.Key("priority"), new Filter.Value(1));
			Filter.Expression typeFilter = new Filter.Expression(Filter.ExpressionType.EQ, new Filter.Key("type"),
					new Filter.Value("A"));
			Filter.Expression complexFilter = new Filter.Expression(Filter.ExpressionType.AND, typeFilter,
					priorityFilter);

			// Reactive: Delete with complex filter
			StepVerifier.create(vectorStore.delete(complexFilter)).expectComplete().verify(TIMEOUT);

			// Reactive: Verify remaining documents
			StepVerifier
				.create(vectorStore
					.similaritySearch(SearchRequest.builder().query("Content").topK(5).similarityThresholdAll().build())
					.collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(2);
					assertThat(results.stream().map(doc -> doc.getMetadata().get("type")).collect(Collectors.toList()))
						.containsExactlyInAnyOrder("A", "B");
					assertThat(
							results.stream().map(doc -> doc.getMetadata().get("priority")).collect(Collectors.toList()))
						.containsExactlyInAnyOrder(1L, 1L);
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Clean up remaining documents
			StepVerifier.create(vectorStore.delete(List.of(doc1.getId(), doc3.getId())))
				.expectComplete()
				.verify(TIMEOUT);
		});
	}

	@Test
	void getNativeClientTest() {
		this.contextRunner.run(context -> {
			QdrantVectorStore vectorStore = context.getBean(QdrantVectorStore.class);
			Optional<QdrantClient> nativeClient = vectorStore.getNativeClient();
			assertThat(nativeClient).isPresent();
		});
	}

	@Test
	void shouldConvertLongToString() {
		this.contextRunner.run(context -> {
			QdrantVectorStore vectorStore = context.getBean(QdrantVectorStore.class);
			var refId = System.currentTimeMillis();
			var doc = new Document("Long type ref_id", Map.of("ref_id", refId));

			// Reactive: Add document
			StepVerifier.create(vectorStore.add(doc)).expectComplete().verify(TIMEOUT);

			// Reactive: Search and verify type conversion
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.builder().query("Long type ref_id").topK(1).build())
					.collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					Document resultDoc = results.get(0);
					var resultRefId = resultDoc.getMetadata().get("ref_id");
					assertThat(resultRefId).isInstanceOf(String.class);
					assertThat(Double.valueOf((String) resultRefId)).isEqualTo(refId);
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Clean up
			StepVerifier.create(vectorStore.delete(List.of(doc.getId()))).expectComplete().verify(TIMEOUT);
		});
	}

	@Test
	void testNonTextDocuments() {
		this.contextRunner.run(context -> {
			QdrantVectorStore vectorStore = context.getBean(QdrantVectorStore.class);
			Media media = new Media(MimeType.valueOf("image/png"), new ByteArrayResource(new byte[] { 0x00 }));

			Document imgDoc = Document.builder().media(media).metadata(Map.of("fileName", "pixel.png")).build();

			// Reactive: Test error handling for non-text documents
			StepVerifier.create(vectorStore.add(imgDoc))
				.expectErrorMatches(throwable -> throwable instanceof IllegalArgumentException && throwable.getMessage()
					.equals("Only text documents are supported for now. One of the documents contains non-text content."))
				.verify(TIMEOUT);
		});
	}

	@Test
	void testCountOperations() {
		this.contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);

			// Reactive: Add documents
			StepVerifier.create(vectorStore.add(this.documents)).expectComplete().verify(TIMEOUT);

			// Reactive: Count all documents
			StepVerifier.create(vectorStore.count())
				.assertNext(count -> assertThat(count).isEqualTo(3))
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Count with filter
			Filter.Expression filter = new Filter.Expression(Filter.ExpressionType.EQ, new Filter.Key("meta1"),
					new Filter.Value("meta1"));

			StepVerifier.create(vectorStore.count(filter))
				.assertNext(count -> assertThat(count).isEqualTo(1))
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Clean up
			StepVerifier.create(vectorStore.delete(this.documents.stream().map(Document::getId).toList()))
				.expectComplete()
				.verify(TIMEOUT);
		});
	}

	@Test
	void testHealthCheck() {
		this.contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);

			// Reactive: Test health check
			StepVerifier.create(vectorStore.isHealthy())
				.assertNext(isHealthy -> assertThat(isHealthy).isTrue())
				.expectComplete()
				.verify(TIMEOUT);
		});
	}

	@Test
	void testSimilaritySearchStream() {
		this.contextRunner.run(context -> {
			VectorStore vectorStore = context.getBean(VectorStore.class);

			// Reactive: Add documents
			StepVerifier.create(vectorStore.add(this.documents)).expectComplete().verify(TIMEOUT);

			// Reactive: Test streaming search
			StepVerifier
				.create(vectorStore.similaritySearchStream(SearchRequest.builder().query("Great").topK(1).build())
					.collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					Document resultDoc = results.get(0);
					assertThat(resultDoc.getId()).isEqualTo(this.documents.get(2).getId());
					assertThat(resultDoc.getText()).isEqualTo(
							"Great Depression Great Depression Great Depression Great Depression Great Depression Great Depression");
				})
				.expectComplete()
				.verify(TIMEOUT);

			// Reactive: Clean up
			StepVerifier.create(vectorStore.delete(this.documents.stream().map(Document::getId).toList()))
				.expectComplete()
				.verify(TIMEOUT);
		});
	}

	@SpringBootConfiguration
	public static class TestApplication {

		@Bean
		public QdrantClient qdrantClient() {
			String host = qdrantContainer.getHost();
			int port = qdrantContainer.getGrpcPort();
			return new QdrantClient(QdrantGrpcClient.newBuilder(host, port, false).build());
		}

		@Bean
		public VectorStore qdrantVectorStore(EmbeddingModel embeddingModel, QdrantClient qdrantClient) {
			return QdrantVectorStore.builder(qdrantClient, embeddingModel)
				.collectionName(COLLECTION_NAME)
				.initializeSchema(true)
				.build();
		}

		@Bean
		public EmbeddingModel embeddingModel() {
			return new OpenAiEmbeddingModel(OpenAiApi.builder().apiKey(System.getenv("OPENAI_API_KEY")).build());
		}

	}

}