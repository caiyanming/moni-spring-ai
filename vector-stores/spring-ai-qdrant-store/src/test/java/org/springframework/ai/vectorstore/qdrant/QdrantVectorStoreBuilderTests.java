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

package org.springframework.ai.vectorstore.qdrant;

import io.qdrant.client.QdrantClient;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import org.springframework.ai.embedding.EmbeddingModel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.Mockito.mock;

/**
 * Tests for {@link QdrantVectorStore.Builder}.
 *
 * @author Mark Pollack
 * @author MoniAI Team
 */
class QdrantVectorStoreBuilderTests {

	private QdrantClient qdrantClient;

	private EmbeddingModel embeddingModel;

	@BeforeEach
	void setUp() {
		this.qdrantClient = mock(QdrantClient.class);
		this.embeddingModel = mock(EmbeddingModel.class);
	}

	@Test
	void defaultConfiguration() {
		QdrantVectorStore vectorStore = QdrantVectorStore.builder(this.qdrantClient, this.embeddingModel).build();

		// Verify default values using the actual field names from our implementation
		assertThat(vectorStore).hasFieldOrPropertyWithValue("collectionName",
				QdrantVectorStore.DEFAULT_COLLECTION_NAME);
		assertThat(vectorStore).hasFieldOrPropertyWithValue("initializeSchema", false);
	}

	@Test
	void customConfiguration() {
		QdrantVectorStore vectorStore = QdrantVectorStore.builder(this.qdrantClient, this.embeddingModel)
			.collectionName("custom_collection")
			.initializeSchema(true)
			.build();

		assertThat(vectorStore).hasFieldOrPropertyWithValue("collectionName", "custom_collection");
		assertThat(vectorStore).hasFieldOrPropertyWithValue("initializeSchema", true);
	}

	@Test
	void nullQdrantClientInConstructorShouldThrowException() {
		assertThatThrownBy(() -> QdrantVectorStore.builder(null, this.embeddingModel))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessage("QdrantClient must not be null");
	}

	@Test
	void nullEmbeddingModelShouldThrowException() {
		assertThatThrownBy(() -> QdrantVectorStore.builder(this.qdrantClient, null))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessage("EmbeddingModel must be configured");
	}

	@Test
	void emptyCollectionNameShouldThrowException() {
		assertThatThrownBy(
				() -> QdrantVectorStore.builder(this.qdrantClient, this.embeddingModel).collectionName("").build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessage("collectionName must not be empty");
	}

	@Test
	void nullCollectionNameShouldThrowException() {
		assertThatThrownBy(
				() -> QdrantVectorStore.builder(this.qdrantClient, this.embeddingModel).collectionName(null).build())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessage("collectionName must not be empty");
	}

	@Test
	void builderMethodChaining() {
		// Verify that all builder methods return the builder instance for method chaining
		QdrantVectorStore.Builder builder = QdrantVectorStore.builder(this.qdrantClient, this.embeddingModel);

		QdrantVectorStore.Builder result = builder.collectionName("test").initializeSchema(true);

		assertThat(result).isSameAs(builder);

		QdrantVectorStore vectorStore = result.build();
		assertThat(vectorStore).isNotNull();
	}

}