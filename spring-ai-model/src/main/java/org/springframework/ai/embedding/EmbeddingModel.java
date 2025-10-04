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

package org.springframework.ai.embedding;

import java.util.ArrayList;
import java.util.List;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import org.springframework.ai.document.Document;
import org.springframework.ai.model.Model;
import org.springframework.util.Assert;

/**
 * EmbeddingModel is a generic interface for embedding models.
 *
 * @author Mark Pollack
 * @author Christian Tzolov
 * @author Josh Long
 * @author Soby Chacko
 * @author Jihoon Kim
 * @since 1.0.0
 *
 */
public interface EmbeddingModel extends Model<EmbeddingRequest, EmbeddingResponse> {

	@Override
	Mono<EmbeddingResponse> call(EmbeddingRequest request);

	/**
	 * Embeds the given text into a vector.
	 * @param text the text to embed.
	 * @return the embedded vector.
	 */
	default Mono<float[]> embed(String text) {
		Assert.notNull(text, "Text must not be null");
		return embed(List.of(text)).map(results -> results.iterator().next());
	}

	/**
	 * Embeds the given document's content into a vector.
	 * @param document the document to embed.
	 * @return the embedded vector.
	 */
	Mono<float[]> embed(Document document);

	/**
	 * Embeds a batch of texts into vectors.
	 * @param texts list of texts to embed.
	 * @return Mono containing list of embedded vectors.
	 */
	default Mono<List<float[]>> embed(List<String> texts) {
		Assert.notNull(texts, "Texts must not be null");
		return this.call(new EmbeddingRequest(texts, EmbeddingOptionsBuilder.builder().build()))
			.map(response -> response.getResults().stream().map(Embedding::getOutput).toList());
	}

	/**
	 * Embeds a batch of {@link Document}s into vectors based on a
	 * {@link BatchingStrategy}.
	 * @param documents list of {@link Document}s.
	 * @param options {@link EmbeddingOptions}.
	 * @param batchingStrategy {@link BatchingStrategy}.
	 * @return Mono containing a list of float[] that represents the vectors for the
	 * incoming {@link Document}s. The returned list is expected to be in the same order
	 * of the {@link Document} list.
	 */
	default Mono<List<float[]>> embed(List<Document> documents, EmbeddingOptions options,
			BatchingStrategy batchingStrategy) {
		Assert.notNull(documents, "Documents must not be null");
		List<List<Document>> batch = batchingStrategy.batch(documents);
		return Flux.fromIterable(batch).flatMapSequential(subBatch -> {
			List<String> texts = subBatch.stream().map(Document::getText).toList();
			EmbeddingRequest request = new EmbeddingRequest(texts, options);
			return this.call(request)
				.map(response -> response.getResults().stream().map(Embedding::getOutput).toList());
		}).collectList().map(batches -> {
			List<float[]> embeddings = new ArrayList<>();
			batches.forEach(embeddings::addAll);
			Assert.isTrue(embeddings.size() == documents.size(),
					"Embeddings must have the same number as that of the documents");
			return embeddings;
		});
	}

	/**
	 * Embeds a batch of texts into vectors and returns the {@link EmbeddingResponse}.
	 * @param texts list of texts to embed.
	 * @return Mono containing the embedding response.
	 */
	default Mono<EmbeddingResponse> embedForResponse(List<String> texts) {
		Assert.notNull(texts, "Texts must not be null");
		return this.call(new EmbeddingRequest(texts, EmbeddingOptionsBuilder.builder().build()));
	}

	/**
	 * Get the number of dimensions of the embedded vectors.
	 *
	 * This is a synchronous method because dimensions are a fixed property of the
	 * embedding model and should not require asynchronous computation. Implementations
	 * should cache this value during initialization to avoid repeated lookups.
	 * @return The number of dimensions of the embedded vectors.
	 * @since 2.1.0
	 */
	int dimensions();

	/**
	 * Get the number of dimensions of the embedded vectors (reactive version).
	 * @deprecated Use {@link #dimensions()} instead. This method is deprecated to avoid
	 * blocking calls in reactive contexts. The new synchronous version expects
	 * implementations to cache dimensions during initialization.
	 * @return Mono containing the number of dimensions of the embedded vectors.
	 */
	@Deprecated(since = "2.1.0", forRemoval = true)
	default Mono<Integer> dimensionsReactive() {
		return Mono.just(dimensions());
	}

}
