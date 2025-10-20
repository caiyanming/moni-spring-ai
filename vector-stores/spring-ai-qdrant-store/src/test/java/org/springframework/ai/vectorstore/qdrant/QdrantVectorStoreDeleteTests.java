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

import io.qdrant.client.QdrantClient;
import io.qdrant.client.grpc.Points.PointsOperationResponse;
import io.qdrant.client.grpc.Points.UpdateResult;
import io.qdrant.client.grpc.Points.UpdateStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.filter.Filter;
import reactor.core.publisher.Mono;
import reactor.test.StepVerifier;

import java.time.Duration;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Unit tests for {@link QdrantVectorStore} deletion operations. Tests the state machine
 * for handling Qdrant's async deletion responses.
 *
 * @author MoniAI Team
 */
class QdrantVectorStoreDeleteTests {

	private QdrantClient qdrantClient;

	private EmbeddingModel embeddingModel;

	private QdrantVectorStore vectorStore;

	@BeforeEach
	void setUp() {
		this.qdrantClient = mock(QdrantClient.class);
		this.embeddingModel = mock(EmbeddingModel.class);
		this.vectorStore = QdrantVectorStore.builder(this.qdrantClient, this.embeddingModel)
			.collectionName("test_collection")
			.build();
	}

	/**
	 * Scenario 1: Normal deletion - UpdateStatus.Completed Expected: Success immediately,
	 * no countPoints() call
	 */
	@Test
	void deleteByFilter_whenCompleted_shouldSucceedImmediately() {
		// Arrange
		Filter.Expression filterExpression = new Filter.Expression(Filter.ExpressionType.EQ, new Filter.Key("status"),
				new Filter.Value("archived"));

		PointsOperationResponse response = PointsOperationResponse.newBuilder()
			.setResult(UpdateResult.newBuilder().setStatus(UpdateStatus.Completed).setOperationId(123).build())
			.setTime(0.5)
			.build();

		when(qdrantClient.deleteByFilter(eq("test_collection"), any(io.qdrant.client.grpc.Points.Filter.class),
				eq(true)))
			.thenReturn(Mono.just(response));

		// Act & Assert
		StepVerifier.create(vectorStore.doDelete(filterExpression)).expectComplete().verify(Duration.ofSeconds(5));

		// Verify countPoints was NOT called
		verify(qdrantClient, never()).countPoints(anyString(), any());
	}

	/**
	 * Scenario 2: Timeout scenario - UpdateStatus.Acknowledged with eventual success
	 * Expected: Retry verification, succeed when countPoints returns 0
	 */
	@Test
	void deleteByFilter_whenAcknowledgedAndEventuallyComplete_shouldRetryAndSucceed() {
		// Arrange
		Filter.Expression filterExpression = new Filter.Expression(Filter.ExpressionType.EQ, new Filter.Key("status"),
				new Filter.Value("archived"));

		PointsOperationResponse response = PointsOperationResponse.newBuilder()
			.setResult(UpdateResult.newBuilder().setStatus(UpdateStatus.Acknowledged).setOperationId(456).build())
			.setTime(0.5)
			.build();

		when(qdrantClient.deleteByFilter(eq("test_collection"), any(io.qdrant.client.grpc.Points.Filter.class),
				eq(true)))
			.thenReturn(Mono.just(response));

		// First countPoints call returns 100 (still deleting)
		// Second countPoints call returns 0 (deletion complete)
		when(qdrantClient.countPoints(eq("test_collection"), any(io.qdrant.client.grpc.Points.Filter.class)))
			.thenReturn(Mono.just(100L))
			.thenReturn(Mono.just(0L));

		// Act & Assert
		StepVerifier.create(vectorStore.doDelete(filterExpression)).expectComplete().verify(Duration.ofSeconds(10));

		// Verify countPoints was called at least twice
		verify(qdrantClient, atLeast(2)).countPoints(eq("test_collection"),
				any(io.qdrant.client.grpc.Points.Filter.class));
	}

	/**
	 * Scenario 3: Timeout scenario - UpdateStatus.Acknowledged with persistent failure
	 * Expected: Retry exhausted after 25 attempts, throw IllegalStateException
	 */
	@Test
	void deleteByFilter_whenAcknowledgedAndNeverComplete_shouldFailAfterRetries() {
		// Arrange
		Filter.Expression filterExpression = new Filter.Expression(Filter.ExpressionType.EQ, new Filter.Key("status"),
				new Filter.Value("archived"));

		PointsOperationResponse response = PointsOperationResponse.newBuilder()
			.setResult(UpdateResult.newBuilder().setStatus(UpdateStatus.Acknowledged).setOperationId(789).build())
			.setTime(0.5)
			.build();

		when(qdrantClient.deleteByFilter(eq("test_collection"), any(io.qdrant.client.grpc.Points.Filter.class),
				eq(true)))
			.thenReturn(Mono.just(response));

		// countPoints always returns 100 (deletion never completes)
		when(qdrantClient.countPoints(eq("test_collection"), any(io.qdrant.client.grpc.Points.Filter.class)))
			.thenReturn(Mono.just(100L));

		// Act & Assert
		StepVerifier.create(vectorStore.doDelete(filterExpression)).expectErrorMatches(error ->
		// Reactor throws RetryExhaustedException when max retries exceeded
		error.getClass().getName().contains("RetryExhaustedException")
				&& error.getMessage().contains("Retries exhausted: 25/25"))
			.verify(Duration.ofSeconds(15));

		// Verify countPoints was called 25 times (initial + 24 retries)
		// Note: Initial call + DELETION_VERIFICATION_MAX_RETRIES (25) = 26 total attempts
		verify(qdrantClient, times(26)).countPoints(eq("test_collection"),
				any(io.qdrant.client.grpc.Points.Filter.class));
	}

	/**
	 * Scenario 4: Unexpected status - UpdateStatus.ClockRejected Expected: Immediate
	 * failure with IllegalStateException
	 */
	@Test
	void deleteByFilter_whenUnexpectedStatus_shouldFailImmediately() {
		// Arrange
		Filter.Expression filterExpression = new Filter.Expression(Filter.ExpressionType.EQ, new Filter.Key("status"),
				new Filter.Value("archived"));

		PointsOperationResponse response = PointsOperationResponse.newBuilder()
			.setResult(UpdateResult.newBuilder().setStatus(UpdateStatus.ClockRejected).setOperationId(999).build())
			.setTime(0.5)
			.build();

		when(qdrantClient.deleteByFilter(eq("test_collection"), any(io.qdrant.client.grpc.Points.Filter.class),
				eq(true)))
			.thenReturn(Mono.just(response));

		// Act & Assert
		StepVerifier.create(vectorStore.doDelete(filterExpression))
			.expectErrorMatches(error -> error instanceof IllegalStateException
					&& error.getMessage().contains("Failed to delete documents by filter")
					&& error.getMessage().contains("ClockRejected"))
			.verify(Duration.ofSeconds(5));

		// Verify countPoints was NOT called
		verify(qdrantClient, never()).countPoints(anyString(), any());
	}

	/**
	 * Scenario 5: Acknowledged with quick completion (first countPoints returns 0)
	 * Expected: Success without retry
	 */
	@Test
	void deleteByFilter_whenAcknowledgedButImmediatelyComplete_shouldSucceedWithoutRetry() {
		// Arrange
		Filter.Expression filterExpression = new Filter.Expression(Filter.ExpressionType.EQ, new Filter.Key("status"),
				new Filter.Value("archived"));

		PointsOperationResponse response = PointsOperationResponse.newBuilder()
			.setResult(UpdateResult.newBuilder().setStatus(UpdateStatus.Acknowledged).setOperationId(111).build())
			.setTime(0.5)
			.build();

		when(qdrantClient.deleteByFilter(eq("test_collection"), any(io.qdrant.client.grpc.Points.Filter.class),
				eq(true)))
			.thenReturn(Mono.just(response));

		// First countPoints call returns 0 (deletion already complete)
		when(qdrantClient.countPoints(eq("test_collection"), any(io.qdrant.client.grpc.Points.Filter.class)))
			.thenReturn(Mono.just(0L));

		// Act & Assert
		StepVerifier.create(vectorStore.doDelete(filterExpression)).expectComplete().verify(Duration.ofSeconds(5));

		// Verify countPoints was called exactly once
		verify(qdrantClient, times(1)).countPoints(eq("test_collection"),
				any(io.qdrant.client.grpc.Points.Filter.class));
	}

}
