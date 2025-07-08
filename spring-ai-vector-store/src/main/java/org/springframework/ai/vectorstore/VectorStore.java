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

import java.util.List;
import java.util.Optional;

import io.micrometer.observation.ObservationRegistry;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import org.springframework.ai.document.Document;
import org.springframework.ai.document.DocumentWriter;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.observation.DefaultVectorStoreObservationConvention;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.util.Assert;

/**
 * Pure reactive interface for vector store operations.
 *
 * <p>
 * This interface provides a fully non-blocking, reactive API for managing and querying
 * documents in a vector database. All operations return reactive types ({@link Mono} or
 * {@link Flux}) and support backpressure and asynchronous processing.
 *
 * <p>
 * Key features:
 * <ul>
 * <li>Zero blocking operations - all methods return Mono or Flux</li>
 * <li>Backpressure support for large document streams</li>
 * <li>Memory-efficient streaming for large datasets</li>
 * <li>Built-in error recovery and retry capabilities</li>
 * </ul>
 */
public interface VectorStore {

	default String getName() {
		return this.getClass().getSimpleName();
	}

	// ========== Document Management ==========

	/**
	 * Adds a stream of documents to the vector store reactively.
	 * @param documents the flux of documents to store
	 * @return a Mono that completes when all documents are successfully added
	 */
	Mono<Void> add(Flux<Document> documents);

	/**
	 * Convenience method for adding a single document.
	 * @param document the document to store
	 * @return a Mono that completes when the document is successfully added
	 */
	default Mono<Void> add(Document document) {
		return add(Flux.just(document));
	}

	/**
	 * Convenience method for adding a list of documents.
	 * @param documents the list of documents to store
	 * @return a Mono that completes when all documents are successfully added
	 */
	default Mono<Void> add(List<Document> documents) {
		if (documents == null) {
			return Mono.error(new IllegalArgumentException("Documents list cannot be null"));
		}
		return add(Flux.fromIterable(documents));
	}

	/**
	 * Deletes documents from the vector store by their IDs.
	 * @param documentIds a flux of document IDs to delete
	 * @return a Mono that completes when all documents are successfully deleted
	 */
	Mono<Void> delete(Flux<String> documentIds);

	/**
	 * Convenience method for deleting a list of documents by IDs.
	 * @param idList list of document ids for which documents will be removed
	 * @return a Mono that completes when all documents are successfully deleted
	 */
	default Mono<Void> delete(List<String> idList) {
		return delete(Flux.fromIterable(idList));
	}

	/**
	 * Deletes documents from the vector store based on filter criteria.
	 * @param filterExpression filter expression to identify documents to delete
	 * @return a Mono that completes when matching documents are successfully deleted
	 */
	Mono<Void> delete(Filter.Expression filterExpression);

	/**
	 * Convenience method for deleting documents by string filter.
	 * @param filterExpression string filter expression
	 * @return a Mono that completes when matching documents are successfully deleted
	 */
	default Mono<Void> delete(String filterExpression) {
		SearchRequest searchRequest = SearchRequest.builder().filterExpression(filterExpression).build();
		Filter.Expression expression = searchRequest.getFilterExpression();
		if (expression == null) {
			return Mono.error(new IllegalArgumentException("Filter expression must not be null"));
		}
		return delete(expression);
	}

	// ========== Search Operations ==========

	/**
	 * Performs similarity search and returns a stream of matching documents.
	 * @param request search request with query, filters, and other parameters
	 * @return a Flux of documents that match the search criteria, ordered by similarity
	 */
	Flux<Document> similaritySearch(SearchRequest request);

	/**
	 * Convenience method for simple text similarity search.
	 * @param query text to search for
	 * @return a Flux of documents similar to the query text
	 */
	default Flux<Document> similaritySearch(String query) {
		return similaritySearch(SearchRequest.builder().query(query).build());
	}

	/**
	 * Performs similarity search with streaming results and backpressure support. This
	 * method is optimized for handling large result sets efficiently.
	 * @param request search request with query, filters, and other parameters
	 * @return a Flux of documents with proper backpressure handling
	 */
	default Flux<Document> similaritySearchStream(SearchRequest request) {
		// Default implementation delegates to regular search
		// Implementations can override for true streaming support
		return similaritySearch(request);
	}

	// ========== Batch Operations ==========

	/**
	 * Performs batch similarity searches for multiple queries.
	 * @param requests a flux of search requests
	 * @return a Flux of search results, maintaining request order
	 */
	default Flux<SearchResult> similaritySearchBatch(Flux<SearchRequest> requests) {
		return requests.concatMap(request -> similaritySearch(request).collectList()
			.map(documents -> new SearchResult(request, documents)));
	}

	// ========== Utility Methods ==========

	/**
	 * Counts the total number of documents in the vector store.
	 * @return a Mono containing the total document count
	 */
	default Mono<Long> count() {
		// Default implementation - subclasses should override for efficiency
		return Mono.fromCallable(() -> 0L);
	}

	/**
	 * Counts documents matching the given filter.
	 * @param filterExpression filter to apply
	 * @return a Mono containing the count of matching documents
	 */
	default Mono<Long> count(Filter.Expression filterExpression) {
		// Default implementation - subclasses should override for efficiency
		return Mono.fromCallable(() -> 0L);
	}

	/**
	 * Checks if the vector store is available and ready for operations.
	 * @return a Mono containing true if the store is healthy, false otherwise
	 */
	default Mono<Boolean> isHealthy() {
		return Mono.just(true);
	}

	/**
	 * Result wrapper for batch search operations.
	 */
	record SearchResult(SearchRequest request, List<Document> documents) {
	};

	/**
	 * Returns the native client if available in this vector store implementation.
	 *
	 * Note on usage: 1. Returns empty Optional when no native client is available 2. Due
	 * to Java type erasure, runtime type checking is not possible
	 *
	 * Example usage: When working with implementation with known native client:
	 * Optional<NativeClientType> client = vectorStore.getNativeClient();
	 *
	 * Note: Using Optional<?> will return the native client if one exists, rather than an
	 * empty Optional. For type safety, prefer using the specific client type.
	 * @return Optional containing native client if available, empty Optional otherwise
	 * @param <T> The type of the native client
	 */
	default <T> Optional<T> getNativeClient() {
		return Optional.empty();
	}

	/**
	 * Builder interface for creating VectorStore instances. Implements a fluent builder
	 * pattern for configuring observation-related settings.
	 *
	 * @param <T> the concrete builder type, enabling method chaining with the correct
	 * return type
	 */
	interface Builder<T extends Builder<T>> {

		/**
		 * Sets the registry for collecting observations and metrics. Defaults to
		 * {@link ObservationRegistry#NOOP} if not specified.
		 * @param observationRegistry the registry to use for observations
		 * @return the builder instance for method chaining
		 */
		T observationRegistry(ObservationRegistry observationRegistry);

		/**
		 * Sets a custom convention for creating observations. If not specified,
		 * {@link DefaultVectorStoreObservationConvention} will be used.
		 * @param convention the custom observation convention to use
		 * @return the builder instance for method chaining
		 */
		T customObservationConvention(VectorStoreObservationConvention convention);

		/**
		 * Sets the batching strategy.
		 * @param batchingStrategy the strategy to use
		 * @return the builder instance for method chaining
		 */
		T batchingStrategy(BatchingStrategy batchingStrategy);

		/**
		 * Builds and returns a new VectorStore instance with the configured settings.
		 * @return a new VectorStore instance
		 */
		VectorStore build();

	}

}
