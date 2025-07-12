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

package org.springframework.ai.vectorstore.qdrant.autoconfigure;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;

import io.grpc.Grpc;
import io.grpc.ManagedChannel;
import io.grpc.TlsChannelCredentials;
import io.qdrant.client.ApiKeyCredentials;
import io.qdrant.client.QdrantGrpcClient;
import io.qdrant.client.grpc.Collections.Distance;
import io.qdrant.client.grpc.Collections.VectorParams;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import reactor.test.StepVerifier;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.transformers.TransformersEmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.DefaultResourceLoader;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Test using a free tier Qdrant Cloud instance: https://cloud.qdrant.io
 *
 * @author Christian Tzolov
 * @author Soby Chacko
 * @since 0.8.1
 */
// NOTE: The free Qdrant Cluster and the QDRANT_API_KEY expire after 4 weeks of
// inactivity.
@EnabledIfEnvironmentVariable(named = "QDRANT_API_KEY", matches = ".+")
@EnabledIfEnvironmentVariable(named = "QDRANT_HOST", matches = ".+")
public class QdrantVectorStoreCloudAutoConfigurationIT {

	private static final String COLLECTION_NAME = "test_collection";

	// Because we pre-create the collection.
	private static final int EMBEDDING_DIMENSION = 384;

	private static final String CLOUD_API_KEY = System.getenv("QDRANT_API_KEY");

	private static final String CLOUD_HOST = System.getenv("QDRANT_HOST");

	// NOTE: The GRPC port (usually 6334) is different from the HTTP port (usually 6333)!
	private static final int CLOUD_GRPC_PORT = 6334;

	private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
		.withConfiguration(AutoConfigurations.of(QdrantVectorStoreAutoConfiguration.class))
		.withUserConfiguration(Config.class)
		.withPropertyValues("spring.ai.vectorstore.qdrant.port=" + CLOUD_GRPC_PORT,
				"spring.ai.vectorstore.qdrant.host=" + CLOUD_HOST,
				"spring.ai.vectorstore.qdrant.api-key=" + CLOUD_API_KEY,
				"spring.ai.vectorstore.qdrant.collection-name=" + COLLECTION_NAME,
				"spring.ai.vectorstore.qdrant.initializeSchema=true", "spring.ai.vectorstore.qdrant.use-tls=true");

	List<Document> documents = List.of(
			new Document(getText("classpath:/test/data/spring.ai.txt"), Map.of("spring", "great")),
			new Document(getText("classpath:/test/data/time.shelter.txt")),
			new Document(getText("classpath:/test/data/great.depression.txt"), Map.of("depression", "bad")));

	@BeforeAll
	static void setup() throws InterruptedException, ExecutionException {

		// Create a new test collection using corrected Qdrant 1.14.1 API
		String hostPort = CLOUD_HOST + ":" + CLOUD_GRPC_PORT;
		ManagedChannel channel = Grpc.newChannelBuilder(hostPort, TlsChannelCredentials.create()).build();

		QdrantGrpcClient client = QdrantGrpcClient.newBuilder()
			.channel(channel)
			.callCredentials(new ApiKeyCredentials(CLOUD_API_KEY))
			.build();

		try {
			// 使用响应式API检查和删除现有集合
			boolean collectionExists = client.listCollections()
				.any(description -> description.getName().equals(COLLECTION_NAME))
				.block();

			if (collectionExists) {
				client.deleteCollection(COLLECTION_NAME).block();
			}

			// 创建新集合
			var vectorParams = VectorParams.newBuilder()
				.setDistance(Distance.Cosine)
				.setSize(EMBEDDING_DIMENSION)
				.build();

			var createRequest = io.qdrant.client.grpc.Collections.CreateCollection.newBuilder()
				.setCollectionName(COLLECTION_NAME)
				.setVectorsConfig(
						io.qdrant.client.grpc.Collections.VectorsConfig.newBuilder().setParams(vectorParams).build())
				.build();

			client.createCollection(createRequest).block();
		}
		finally {
			client.close();
			channel.shutdown();
		}
	}

	public static String getText(String uri) {
		var resource = new DefaultResourceLoader().getResource(uri);
		try {
			return resource.getContentAsString(StandardCharsets.UTF_8);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Test
	public void addAndSearch() {
		this.contextRunner.run(context -> {

			VectorStore vectorStore = context.getBean(VectorStore.class);

			// 纯响应式测试：添加文档
			StepVerifier.create(vectorStore.add(this.documents)).verifyComplete();

			// 纯响应式测试：相似性搜索
			StepVerifier
				.create(vectorStore
					.similaritySearch(SearchRequest.builder().query("What is Great Depression?").topK(1).build())
					.collectList())
				.assertNext(results -> {
					assertThat(results).hasSize(1);
					Document resultDoc = results.get(0);
					assertThat(resultDoc.getId()).isEqualTo(this.documents.get(2).getId());
					assertThat(resultDoc.getMetadata()).containsKeys("depression", "distance");
				})
				.verifyComplete();

			// 纯响应式测试：删除文档
			StepVerifier.create(vectorStore.delete(this.documents.stream().map(Document::getId).toList()))
				.verifyComplete();

			// 纯响应式测试：验证删除结果
			StepVerifier
				.create(vectorStore.similaritySearch(SearchRequest.builder().query("Great Depression").topK(1).build())
					.collectList())
				.assertNext(results -> assertThat(results).hasSize(0))
				.verifyComplete();
		});
	}

	@Configuration(proxyBeanMethods = false)
	static class Config {

		@Bean
		public EmbeddingModel embeddingModel() {
			return new TransformersEmbeddingModel();
		}

	}

}
