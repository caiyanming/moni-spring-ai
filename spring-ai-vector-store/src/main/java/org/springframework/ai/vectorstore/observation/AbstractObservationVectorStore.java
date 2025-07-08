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

package org.springframework.ai.vectorstore.observation;

import java.util.List;

import io.micrometer.observation.ObservationRegistry;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.AbstractVectorStoreBuilder;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.lang.Nullable;

/**
 * Abstract base class for {@link VectorStore} implementations that provides observation
 * capabilities.
 *
 * @author Christian Tzolov
 * @author Soby Chacko
 * @since 1.0.0
 */
public abstract class AbstractObservationVectorStore implements VectorStore {

	private static final VectorStoreObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultVectorStoreObservationConvention();

	private final ObservationRegistry observationRegistry;

	@Nullable
	private final VectorStoreObservationConvention customObservationConvention;

	protected final EmbeddingModel embeddingModel;

	protected final BatchingStrategy batchingStrategy;

	private AbstractObservationVectorStore(EmbeddingModel embeddingModel, ObservationRegistry observationRegistry,
			@Nullable VectorStoreObservationConvention customObservationConvention, BatchingStrategy batchingStrategy) {
		this.embeddingModel = embeddingModel;
		this.observationRegistry = observationRegistry;
		this.customObservationConvention = customObservationConvention;
		this.batchingStrategy = batchingStrategy;
	}

	/**
	 * Creates a new AbstractObservationVectorStore instance with the specified builder
	 * settings. Initializes observation-related components and the embedding model.
	 * @param builder the builder containing configuration settings
	 */
	public AbstractObservationVectorStore(AbstractVectorStoreBuilder<?> builder) {
		this(builder.getEmbeddingModel(), builder.getObservationRegistry(), builder.getCustomObservationConvention(),
				builder.getBatchingStrategy());
	}

	/**
	 * Create a new {@link AbstractObservationVectorStore} instance.
	 * @param documents the documents to add
	 */
	@Override
	public Mono<Void> add(Flux<Document> documents) {
		return documents.collectList().flatMap(docList -> {
			validateNonTextDocuments(docList);
			VectorStoreObservationContext observationContext = this
				.createObservationContextBuilder(VectorStoreObservationContext.Operation.ADD.value())
				.build();

			return VectorStoreObservationDocumentation.AI_VECTOR_STORE
				.observation(this.customObservationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
						this.observationRegistry)
				.observe(() -> this.doAdd(docList));
		});
	}

	private void validateNonTextDocuments(List<Document> documents) {
		if (documents == null)
			return;
		for (Document document : documents) {
			if (document != null && !document.isText()) {
				throw new IllegalArgumentException(
						"Only text documents are supported for now. One of the documents contains non-text content.");
			}
		}
	}

	@Override
	public Mono<Void> delete(Flux<String> deleteDocIds) {
		return deleteDocIds.collectList().flatMap(docIds -> {
			VectorStoreObservationContext observationContext = this
				.createObservationContextBuilder(VectorStoreObservationContext.Operation.DELETE.value())
				.build();

			return VectorStoreObservationDocumentation.AI_VECTOR_STORE
				.observation(this.customObservationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
						this.observationRegistry)
				.observe(() -> this.doDelete(docIds));
		});
	}

	@Override
	public Mono<Void> delete(Filter.Expression filterExpression) {
		VectorStoreObservationContext observationContext = this
			.createObservationContextBuilder(VectorStoreObservationContext.Operation.DELETE.value())
			.build();

		return VectorStoreObservationDocumentation.AI_VECTOR_STORE
			.observation(this.customObservationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> this.doDelete(filterExpression));
	}

	@Override
	public Flux<Document> similaritySearch(SearchRequest request) {

		VectorStoreObservationContext searchObservationContext = this
			.createObservationContextBuilder(VectorStoreObservationContext.Operation.QUERY.value())
			.queryRequest(request)
			.build();

		return this.doSimilaritySearch(request)
			.doOnNext(documents -> searchObservationContext.setQueryResponse(documents))
			.flatMapMany(Flux::fromIterable);
	}

	/**
	 * Perform the actual add operation.
	 * @param documents the documents to add
	 */
	public abstract Mono<Void> doAdd(List<Document> documents);

	/**
	 * Perform the actual delete operation.
	 * @param idList the list of document IDs to delete
	 */
	public abstract Mono<Void> doDelete(List<String> idList);

	/**
	 * Template method for concrete implementations to provide filter-based deletion
	 * logic.
	 * @param filterExpression Filter expression to identify documents to delete
	 */
	protected Mono<Void> doDelete(Filter.Expression filterExpression) {
		// this is temporary until we implement this method in all concrete vector stores,
		// at which point
		// this method will become an abstract method.
		return Mono.error(new UnsupportedOperationException());
	}

	/**
	 * Perform the actual similarity search operation.
	 * @param request the search request
	 * @return a Mono containing the list of documents that match the query request
	 * conditions
	 */
	public abstract Mono<List<Document>> doSimilaritySearch(SearchRequest request);

	/**
	 * Create a new {@link VectorStoreObservationContext.Builder} instance.
	 * @param operationName the operation name
	 * @return the observation context builder
	 */
	public abstract VectorStoreObservationContext.Builder createObservationContextBuilder(String operationName);

}
