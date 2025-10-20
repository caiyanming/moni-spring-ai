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

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ExecutionException;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import io.qdrant.client.QdrantClient;
import io.qdrant.client.grpc.Collections.Distance;
import io.qdrant.client.grpc.Collections.VectorParams;
import io.qdrant.client.grpc.Collections.VectorsConfig;
import io.qdrant.client.grpc.Collections.CreateCollection;
import io.qdrant.client.grpc.Collections.CollectionDescription;
import io.qdrant.client.grpc.JsonWithInt.Value;
import io.qdrant.client.grpc.Points.Filter;
import io.qdrant.client.grpc.Points.PointId;
import io.qdrant.client.grpc.Points.PointStruct;
import io.qdrant.client.grpc.Points.ScoredPoint;
import io.qdrant.client.grpc.Points.SearchPoints;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.ai.document.Document;
import org.springframework.ai.document.DocumentMetadata;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingOptionsBuilder;
import org.springframework.ai.model.EmbeddingUtils;
import org.springframework.ai.observation.conventions.VectorStoreProvider;
import org.springframework.ai.vectorstore.AbstractVectorStoreBuilder;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.observation.AbstractObservationVectorStore;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationContext;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.util.Assert;

/**
 * Qdrant vectorStore implementation. This store supports creating, updating, deleting,
 * and similarity searching of documents in a Qdrant collection.
 *
 * <p>
 * The store uses Qdrant's vector search functionality to persist and query vector
 * embeddings along with their associated document content and metadata. The
 * implementation leverages Qdrant's HNSW (Hierarchical Navigable Small World) algorithm
 * for efficient k-NN search operations.
 * </p>
 *
 * <p>
 * Features:
 * </p>
 * <ul>
 * <li>Automatic schema initialization with configurable collection creation</li>
 * <li>Support for cosine similarity distance metric</li>
 * <li>Metadata filtering using Qdrant's filter expressions</li>
 * <li>Configurable similarity thresholds for search results</li>
 * <li>Batch processing support with configurable strategies</li>
 * <li>Observation and metrics support through Micrometer</li>
 * </ul>
 *
 * <p>
 * Basic usage example:
 * </p>
 * <pre>{@code
 * QdrantVectorStore vectorStore = QdrantVectorStore.builder(qdrantClient)
 *     .embeddingModel(embeddingModel)
 *     .initializeSchema(true)
 *     .build();
 *
 * // Add documents
 * vectorStore.add(List.of(
 *     new Document("content1", Map.of("key1", "value1")),
 *     new Document("content2", Map.of("key2", "value2"))
 * ));
 *
 * // Search with filters
 * List<Document> results = vectorStore.similaritySearch(
 *     SearchRequest.query("search text")
 *         .withTopK(5)
 *         .withSimilarityThreshold(0.7)
 *         .withFilterExpression("key1 == 'value1'")
 * );
 * }</pre>
 *
 * <p>
 * Advanced configuration example:
 * </p>
 * <pre>{@code
 * QdrantVectorStore vectorStore = QdrantVectorStore.builder(qdrantClient, embeddingModel)
 *     .collectionName("custom-collection")
 *     .initializeSchema(true)
 *     .batchingStrategy(new TokenCountBatchingStrategy())
 *     .observationRegistry(observationRegistry)
 *     .customObservationConvention(customConvention)
 *     .build();
 * }</pre>
 *
 * <p>
 * Requirements:
 * </p>
 * <ul>
 * <li>Running Qdrant instance accessible via gRPC</li>
 * <li>Collection with vector size matching the embedding model dimensions</li>
 * </ul>
 *
 * @author Anush Shetty
 * @author Christian Tzolov
 * @author Eddú Meléndez
 * @author Josh Long
 * @author Soby Chacko
 * @author Thomas Vitale
 * @since 1.0.0
 */
public class QdrantVectorStore extends AbstractObservationVectorStore implements InitializingBean {

	private static final Logger logger = LoggerFactory.getLogger(QdrantVectorStore.class);

	public static final String DEFAULT_COLLECTION_NAME = "vector_store";

	private static final String CONTENT_FIELD_NAME = "doc_content";

	private final QdrantClient qdrantClient;

	private final String collectionName;

	private final QdrantFilterExpressionConverter filterExpressionConverter = new QdrantFilterExpressionConverter();

	private final boolean initializeSchema;

	/**
	 * Protected constructor for creating a QdrantVectorStore instance using the builder
	 * pattern.
	 * @param builder the {@link Builder} containing all configuration settings
	 * @throws IllegalArgumentException if qdrant client is missing
	 * @see Builder
	 * @since 1.0.0
	 */
	protected QdrantVectorStore(Builder builder) {
		super(builder);

		Assert.notNull(builder.qdrantClient, "QdrantClient must not be null");

		this.qdrantClient = builder.qdrantClient;
		this.collectionName = builder.collectionName;
		this.initializeSchema = builder.initializeSchema;
	}

	/**
	 * Creates a new QdrantBuilder instance. This is the recommended way to instantiate a
	 * QdrantVectorStore.
	 * @param qdrantClient the client for interfacing with Qdrant
	 * @return a new QdrantBuilder instance
	 */
	public static Builder builder(QdrantClient qdrantClient, EmbeddingModel embeddingModel) {
		return new Builder(qdrantClient, embeddingModel);
	}

	/**
	 * Adds a list of documents to the vector store.
	 * @param documents The list of documents to be added.
	 */
	@Override
	public Mono<Void> doAdd(List<Document> documents) {
		// Compute and assign an embedding to the document.
		return this.embeddingModel.embed(documents, EmbeddingOptionsBuilder.builder().build(), this.batchingStrategy)
			.flatMap(embeddings -> {
				List<PointStruct> points = documents.stream()
					.map(document -> PointStruct.newBuilder()
						.setId(io.qdrant.client.PointIdFactory.id(UUID.fromString(document.getId())))
						.setVectors(
								io.qdrant.client.VectorsFactory.vectors(embeddings.get(documents.indexOf(document))))
						.putAllPayload(toPayload(document))
						.build())
					.toList();

				return this.qdrantClient.upsert(this.collectionName, Flux.fromIterable(points)).then();
			})
			.onErrorMap(Exception.class, e -> new RuntimeException("Failed to add documents", e));
	}

	/**
	 * Deletes a list of documents by their IDs.
	 * @param documentIds The list of document IDs to be deleted.
	 */
	@Override
	public Mono<Void> doDelete(List<String> documentIds) {
		List<PointId> ids = documentIds.stream()
			.map(id -> io.qdrant.client.PointIdFactory.id(UUID.fromString(id)))
			.toList();
		return this.qdrantClient.delete(this.collectionName, Flux.fromIterable(ids))
			.then()
			.onErrorMap(Exception.class, e -> new RuntimeException("Failed to delete documents", e));
	}

	@Override
	protected Mono<Void> doDelete(org.springframework.ai.vectorstore.filter.Filter.Expression filterExpression) {
		Assert.notNull(filterExpression, "Filter expression must not be null");

		Filter filter = this.filterExpressionConverter.convertExpression(filterExpression);

		return this.qdrantClient.deleteByFilter(this.collectionName, filter, true).flatMap(response -> {
			var status = response.getResult().getStatus();

			// Completed: deletion fully applied and ready for search
			if (status == io.qdrant.client.grpc.Points.UpdateStatus.Completed) {
				logger.debug("Deletion completed (status: Completed)");
				return Mono.<Void>empty();
			}

			// Acknowledged: deletion received but not yet processed (timeout scenario)
			// Per Qdrant source (clean.rs:139), wait=true may return Acknowledged on timeout
			// We must verify deletion completed before continuing
			if (status == io.qdrant.client.grpc.Points.UpdateStatus.Acknowledged) {
				logger.warn("Deletion acknowledged but not completed (timeout), verifying deletion state...");
				return verifyDeletionCompleted(filter);
			}

			return Mono.error(new IllegalStateException("Failed to delete documents by filter: " + status));
		}).doOnError(e -> logger.error("Failed to delete documents by filter: {}", e.getMessage(), e));
	}

	/**
	 * Verifies that deletion has completed by checking if any points remain matching the filter.
	 * This is necessary when deleteByFilter returns Acknowledged status (timeout scenario).
	 *
	 * <p>Acknowledged means deletion is "received but not yet processed" - Qdrant is still
	 * deleting in the background. We retry for up to 5 seconds to give Qdrant time to complete.
	 *
	 * @param filter the filter to check
	 * @return Mono that completes successfully if no points remain, or errors after retry exhausted
	 */
	private Mono<Void> verifyDeletionCompleted(Filter filter) {
		return Mono.defer(() -> this.qdrantClient.countPoints(this.collectionName, filter))
			.flatMap(count -> {
				if (count == 0) {
					logger.debug("Deletion verified: no remaining points");
					return Mono.<Void>empty();
				}

				// Points still exist - deletion still in progress
				logger.debug("Deletion still in progress: {} points remaining, retrying...", count);
				return Mono.error(new DeletionStillInProgressException(count));
			})
			.retryWhen(reactor.util.retry.Retry.fixedDelay(25, java.time.Duration.ofMillis(200))
				.filter(throwable -> throwable instanceof DeletionStillInProgressException)
				.doBeforeRetry(signal -> {
					DeletionStillInProgressException ex = (DeletionStillInProgressException) signal.failure();
					logger.debug("Retry {}/{}: {} points remaining",
						signal.totalRetries() + 1, 25, ex.getRemainingCount());
				})
			)
			.onErrorResume(DeletionStillInProgressException.class, ex -> {
				String errorMsg = String.format(
					"Deletion incomplete after 5 seconds of verification: %d points still exist. " +
					"Qdrant background deletion may be slow. Consider increasing timeout or retry later.",
					ex.getRemainingCount()
				);
				logger.error(errorMsg);
				return Mono.error(new IllegalStateException(errorMsg));
			})
			.doOnError(e -> {
				if (!(e instanceof IllegalStateException)) {
					logger.error("Failed to verify deletion state", e);
				}
			});
	}

	/**
	 * Exception indicating deletion is still in progress in Qdrant background.
	 * Used internally for retry logic.
	 */
	private static class DeletionStillInProgressException extends RuntimeException {
		private final long remainingCount;

		DeletionStillInProgressException(long remainingCount) {
			super("Deletion still in progress: " + remainingCount + " points remaining");
			this.remainingCount = remainingCount;
		}

		public long getRemainingCount() {
			return remainingCount;
		}
	}

	/**
	 * Performs a similarity search on the vector store.
	 * @param request The {@link SearchRequest} object containing the query and other
	 * search parameters.
	 * @return A Mono containing the list of documents that are similar to the query.
	 */
	@Override
	public Mono<List<Document>> doSimilaritySearch(SearchRequest request) {
		Filter filter = (request.getFilterExpression() != null)
				? this.filterExpressionConverter.convertExpression(request.getFilterExpression())
				: Filter.getDefaultInstance();

		return this.embeddingModel.embed(request.getQuery()).flatMap(queryEmbedding -> {
			var searchPoints = SearchPoints.newBuilder()
				.setCollectionName(this.collectionName)
				.setLimit(request.getTopK())
				.setWithPayload(io.qdrant.client.WithPayloadSelectorFactory.enable(true))
				.addAllVector(EmbeddingUtils.toList(queryEmbedding))
				.setFilter(filter)
				.setScoreThreshold((float) request.getSimilarityThreshold())
				.build();

			return this.qdrantClient.search(searchPoints).map(this::toDocument).collectList();
		}).onErrorMap(Exception.class, e -> new RuntimeException("Failed to perform similarity search", e));
	}

	/**
	 * Returns {@link Document} using the {@link ScoredPoint}
	 * @param point ScoredPoint containing the query response.
	 * @return the {@link Document} representing the response.
	 */
	private Document toDocument(ScoredPoint point) {
		try {
			var id = point.getId().getUuid();

			var metadata = QdrantObjectFactory.toObjectMap(point.getPayloadMap());
			metadata.put(DocumentMetadata.DISTANCE.value(), 1 - point.getScore());

			var content = (String) metadata.remove(CONTENT_FIELD_NAME);

			return Document.builder().id(id).text(content).metadata(metadata).score((double) point.getScore()).build();
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Converts the document metadata to a Protobuf Struct.
	 * @param document The document containing metadata.
	 * @return The metadata as a Protobuf Struct.
	 */
	private Map<String, Value> toPayload(Document document) {
		try {
			var payload = QdrantValueFactory.toValueMap(document.getMetadata());
			payload.put(CONTENT_FIELD_NAME, io.qdrant.client.ValueFactory.value(document.getText()));
			return payload;
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void afterPropertiesSet() throws Exception {

		// CRITICAL: Initialize dimensions early on main thread to avoid blocking on Netty
		// event loop later. This call may block if model is not in known dimensions
		// registry, but it's safe here as we're in Bean initialization phase.
		int dimensions = this.embeddingModel.dimensions();

		if (!this.initializeSchema) {
			return;
		}

		// Create the collection if it does not exist.
		isCollectionExists().flatMap(exists -> {
			if (!exists) {
				var vectorParams = VectorParams.newBuilder().setDistance(Distance.Cosine).setSize(dimensions).build();
				return this.qdrantClient
					.createCollection(CreateCollection.newBuilder()
						.setCollectionName(this.collectionName)
						.setVectorsConfig(VectorsConfig.newBuilder().setParams(vectorParams).build())
						.build())
					.then();
			}
			return Mono.empty();
		}).block();
	}

	private Mono<Boolean> isCollectionExists() {
		return this.qdrantClient.listCollections()
			.map(CollectionDescription::getName)
			.any(name -> name.equals(this.collectionName))
			.onErrorMap(Exception.class, e -> new RuntimeException("Failed to check collection existence", e));
	}

	@Override
	public VectorStoreObservationContext.Builder createObservationContextBuilder(String operationName) {

		return VectorStoreObservationContext.builder(VectorStoreProvider.QDRANT.value(), operationName)
			.dimensions(this.embeddingModel.dimensions())
			.collectionName(this.collectionName);

	}

	@Override
	public <T> Optional<T> getNativeClient() {
		@SuppressWarnings("unchecked")
		T client = (T) this.qdrantClient;
		return Optional.of(client);
	}

	/**
	 * Builder for creating instances of {@link QdrantVectorStore}. This builder provides
	 * a fluent API for configuring all aspects of the vector store.
	 *
	 * @since 1.0.0
	 */
	public static class Builder extends AbstractVectorStoreBuilder<Builder> {

		private final QdrantClient qdrantClient;

		private String collectionName = DEFAULT_COLLECTION_NAME;

		private boolean initializeSchema = false;

		/**
		 * Creates a new builder instance with the required QdrantClient and
		 * EmbeddingModel.
		 * @param qdrantClient the client for Qdrant operations
		 * @throws IllegalArgumentException if qdrantClient is null
		 */
		private Builder(QdrantClient qdrantClient, EmbeddingModel embeddingModel) {
			super(embeddingModel);
			Assert.notNull(qdrantClient, "QdrantClient must not be null");
			this.qdrantClient = qdrantClient;
		}

		/**
		 * Configures the Qdrant collection name.
		 * @param collectionName the name of the collection to use (defaults to
		 * {@value DEFAULT_COLLECTION_NAME})
		 * @return this builder instance
		 * @throws IllegalArgumentException if collectionName is null or empty
		 */
		public Builder collectionName(String collectionName) {
			Assert.hasText(collectionName, "collectionName must not be empty");
			this.collectionName = collectionName;
			return this;
		}

		/**
		 * Configures whether to initialize the collection schema.
		 * @param initializeSchema true to initialize schema automatically
		 * @return this builder instance
		 */
		public Builder initializeSchema(boolean initializeSchema) {
			this.initializeSchema = initializeSchema;
			return this;
		}

		/**
		 * Builds and returns a new QdrantVectorStore instance with the configured
		 * settings.
		 * @return a new QdrantVectorStore instance
		 * @throws IllegalStateException if the builder configuration is invalid
		 */
		@Override
		public QdrantVectorStore build() {
			return new QdrantVectorStore(this);
		}

	}

}
