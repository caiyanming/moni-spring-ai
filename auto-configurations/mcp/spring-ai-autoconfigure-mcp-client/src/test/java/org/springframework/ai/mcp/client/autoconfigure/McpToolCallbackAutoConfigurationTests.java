/*
 * Copyright 2025-2025 the original author or authors.
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

package org.springframework.ai.mcp.client.autoconfigure;

import java.util.List;
import java.util.function.Function;

import com.fasterxml.jackson.core.type.TypeReference;
import io.modelcontextprotocol.spec.McpClientTransport;
import io.modelcontextprotocol.spec.McpSchema;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import reactor.core.publisher.Mono;

import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.assertj.core.api.Assertions.assertThat;

public class McpToolCallbackAutoConfigurationTests {

	private final ApplicationContextRunner applicationContext = new ApplicationContextRunner().withConfiguration(
			AutoConfigurations.of(McpClientAutoConfiguration.class, McpToolCallbackAutoConfiguration.class));

	@Test
	void enabledByDefault() {

		// Pure reactive mode - always creates async tool callbacks
		this.applicationContext.withUserConfiguration(TestTransportConfiguration.class).run(context -> {
			assertThat(context).hasBean("mcpToolCallbacks");
		});

		// SYNC mode should fail with exception during configuration
		this.applicationContext.withUserConfiguration(TestTransportConfiguration.class)
			.withPropertyValues("spring.ai.mcp.client.enabled=true", "spring.ai.mcp.client.type=SYNC")
			.run(context -> {
				assertThat(context).hasFailed();
				assertThat(context.getStartupFailure())
					.hasMessageContaining("Synchronous MCP client mode is not supported");
			});

		this.applicationContext.withUserConfiguration(TestTransportConfiguration.class)
			.withPropertyValues("spring.ai.mcp.client.enabled=true", "spring.ai.mcp.client.type=ASYNC")
			.run(context -> {
				assertThat(context).hasBean("mcpToolCallbacks");
			});
	}

	@Test
	void enabledMcpToolCallbackAutoConfiguration() {

		// Pure reactive mode - always creates async tool callbacks
		this.applicationContext.withUserConfiguration(TestTransportConfiguration.class)
			.withPropertyValues("spring.ai.mcp.client.toolcallback.enabled=true")
			.run(context -> {
				assertThat(context).hasBean("mcpToolCallbacks");
			});

		// SYNC mode should fail with exception
		this.applicationContext.withUserConfiguration(TestTransportConfiguration.class)
			.withPropertyValues("spring.ai.mcp.client.enabled=true", "spring.ai.mcp.client.toolcallback.enabled=true",
					"spring.ai.mcp.client.type=SYNC")
			.run(context -> {
				assertThat(context).hasFailed();
				assertThat(context.getStartupFailure())
					.hasMessageContaining("Synchronous MCP client mode is not supported");
			});

		// Async mode works fine
		this.applicationContext.withUserConfiguration(TestTransportConfiguration.class)
			.withPropertyValues("spring.ai.mcp.client.toolcallback.enabled=true", "spring.ai.mcp.client.type=ASYNC")
			.run(context -> {
				assertThat(context).hasBean("mcpToolCallbacks");
			});

		this.applicationContext.withUserConfiguration(TestTransportConfiguration.class)
			.withPropertyValues("spring.ai.mcp.client.enabled=true", "spring.ai.mcp.client.toolcallback.enabled=true",
					"spring.ai.mcp.client.type=ASYNC")
			.run(context -> {
				assertThat(context).hasBean("mcpToolCallbacks");
			});
	}

	@Configuration
	static class TestTransportConfiguration {

		@Bean
		List<NamedClientMcpTransport> testTransports() {
			McpClientTransport mockTransport = Mockito.mock(McpClientTransport.class);
			Mockito.when(mockTransport.connect(Mockito.any())).thenReturn(Mono.empty());
			Mockito.when(mockTransport.sendMessage(Mockito.any())).thenReturn(Mono.empty());
			Mockito.when(mockTransport.closeGracefully()).thenReturn(Mono.empty());
			return List.of(new NamedClientMcpTransport("test", mockTransport));
		}

	}

	static class CustomClientTransport implements McpClientTransport {

		@Override
		public void close() {
			// Test implementation
		}

		@Override
		public Mono<Void> connect(
				Function<Mono<McpSchema.JSONRPCMessage>, Mono<McpSchema.JSONRPCMessage>> messageHandler) {
			return Mono.empty(); // Test implementation
		}

		@Override
		public Mono<Void> sendMessage(McpSchema.JSONRPCMessage message) {
			return Mono.empty(); // Test implementation
		}

		@Override
		public <T> T unmarshalFrom(Object value, TypeReference<T> type) {
			return null; // Test implementation
		}

		@Override
		public Mono<Void> closeGracefully() {
			return Mono.empty(); // Test implementation
		}

	}

}
