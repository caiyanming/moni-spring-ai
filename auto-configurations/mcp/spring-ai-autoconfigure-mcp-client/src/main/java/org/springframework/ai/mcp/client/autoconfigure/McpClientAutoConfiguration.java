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

import java.util.ArrayList;
import java.util.List;

import io.modelcontextprotocol.client.McpAsyncClient;
import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.spec.McpSchema;

import org.springframework.ai.mcp.client.autoconfigure.configurer.McpAsyncClientConfigurer;
import org.springframework.ai.mcp.client.autoconfigure.properties.McpClientCommonProperties;
import org.springframework.ai.mcp.customizer.McpAsyncClientCustomizer;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.util.CollectionUtils;

/**
 * Auto-configuration for Model Context Protocol (MCP) client support.
 *
 * <p>
 * This configuration class sets up the necessary beans for MCP client functionality, with
 * pure reactive (asynchronous) clients and their respective tool callbacks. It is
 * automatically enabled when the required classes are present on the classpath and can be
 * explicitly disabled through properties.
 *
 * <p>
 * Configuration Properties:
 * <ul>
 * <li>{@code spring.ai.mcp.client.enabled} - Enable/disable MCP client support (default:
 * true)
 * <li>{@code spring.ai.mcp.client.name} - Client implementation name
 * <li>{@code spring.ai.mcp.client.version} - Client implementation version
 * <li>{@code spring.ai.mcp.client.request-timeout} - Request timeout duration
 * </ul>
 *
 * <p>
 * The configuration is activated after the transport-specific auto-configurations (Stdio,
 * SSE HTTP, and SSE WebFlux) to ensure proper initialization order. At least one
 * transport must be available for the clients to be created.
 *
 * <p>
 * Key features:
 * <ul>
 * <li>Pure Reactive (Async) Client Support:
 * <ul>
 * <li>Creates and configures MCP async clients based on available transports
 * <li>Supports only non-blocking reactive operations
 * <li>Automatic client initialization on application ready event
 * </ul>
 * <li>Integration Support:
 * <ul>
 * <li>Sets up tool callbacks for Spring AI integration
 * <li>Supports multiple named transports
 * <li>Proper lifecycle management with automatic cleanup
 * </ul>
 * <li>Customization Options:
 * <ul>
 * <li>Extensible through {@link McpAsyncClientCustomizer}
 * <li>Configurable timeouts and client information
 * <li>Support for custom transport implementations
 * </ul>
 * </ul>
 *
 * @see McpAsyncClient
 * @see McpClientCommonProperties
 * @see McpAsyncClientCustomizer
 * @see StdioTransportAutoConfiguration
 * @see SseHttpClientTransportAutoConfiguration
 * @see SseWebFluxTransportAutoConfiguration
 */
@AutoConfiguration(after = { StdioTransportAutoConfiguration.class, SseHttpClientTransportAutoConfiguration.class,
		SseWebFluxTransportAutoConfiguration.class })
@ConditionalOnClass({ McpSchema.class })
@EnableConfigurationProperties(McpClientCommonProperties.class)
@ConditionalOnProperty(prefix = McpClientCommonProperties.CONFIG_PREFIX, name = "enabled", havingValue = "true",
		matchIfMissing = true)
public class McpClientAutoConfiguration {

	/**
	 * Create a dynamic client name based on the client name and the name of the server
	 * connection.
	 * @param clientName the client name as defined by the configuration
	 * @param serverConnectionName the name of the server connection being used by the
	 * client
	 * @return the connected client name
	 */
	private String connectedClientName(String clientName, String serverConnectionName) {
		return clientName + " - " + serverConnectionName;
	}

	// Pure Reactive (Async) client configuration

	@Bean
	@ConditionalOnProperty(prefix = McpClientCommonProperties.CONFIG_PREFIX, name = "enabled", havingValue = "true",
			matchIfMissing = true)
	public List<McpAsyncClient> mcpAsyncClients(McpAsyncClientConfigurer mcpAsyncClientConfigurer,
			McpClientCommonProperties commonProperties,
			ObjectProvider<List<NamedClientMcpTransport>> transportsProvider) {

		// Validate that only ASYNC client type is supported in pure reactive mode
		if (commonProperties.getType() == McpClientCommonProperties.ClientType.SYNC) {
			throw new IllegalStateException(
					"Synchronous MCP client mode is not supported in pure reactive configuration. "
							+ "Please set spring.ai.mcp.client.type=ASYNC or remove this property to use the default ASYNC mode.");
		}

		List<McpAsyncClient> mcpAsyncClients = new ArrayList<>();

		List<NamedClientMcpTransport> namedTransports = transportsProvider.stream().flatMap(List::stream).toList();

		if (!CollectionUtils.isEmpty(namedTransports)) {
			for (NamedClientMcpTransport namedTransport : namedTransports) {

				McpSchema.Implementation clientInfo = new McpSchema.Implementation(
						this.connectedClientName(commonProperties.getName(), namedTransport.name()),
						commonProperties.getVersion());

				McpClient.AsyncSpec spec = McpClient.async(namedTransport.transport())
					.clientInfo(clientInfo)
					.requestTimeout(commonProperties.getRequestTimeout());

				spec = mcpAsyncClientConfigurer.configure(namedTransport.name(), spec);

				var client = spec.build();

				mcpAsyncClients.add(client);
			}
		}

		return mcpAsyncClients;
	}

	@Bean
	@ConditionalOnProperty(prefix = McpClientCommonProperties.CONFIG_PREFIX, name = "enabled", havingValue = "true",
			matchIfMissing = true)
	public CloseableMcpAsyncClients makeAsyncClientsClosable(List<McpAsyncClient> clients) {
		return new CloseableMcpAsyncClients(clients);
	}

	@Bean
	@ConditionalOnMissingBean
	@ConditionalOnProperty(prefix = McpClientCommonProperties.CONFIG_PREFIX, name = "enabled", havingValue = "true",
			matchIfMissing = true)
	McpAsyncClientConfigurer mcpAsyncClientConfigurer(ObjectProvider<McpAsyncClientCustomizer> customizerProvider) {
		return new McpAsyncClientConfigurer(customizerProvider.orderedStream().toList());
	}

	/**
	 * Component responsible for initializing MCP async clients after application startup.
	 * This ensures all clients are properly initialized in a reactive manner without
	 * blocking the startup process.
	 */
	@Component
	static class McpClientInitializer {

		private final McpClientCommonProperties commonProperties;

		private final List<McpAsyncClient> mcpAsyncClients;

		public McpClientInitializer(McpClientCommonProperties commonProperties,
				ObjectProvider<List<McpAsyncClient>> mcpAsyncClientsProvider) {
			this.commonProperties = commonProperties;
			this.mcpAsyncClients = mcpAsyncClientsProvider.stream().flatMap(List::stream).toList();
		}

		@EventListener(ApplicationReadyEvent.class)
		public void initializeClientsAsync() {
			if (commonProperties.isInitialized() && !CollectionUtils.isEmpty(mcpAsyncClients)) {
				mcpAsyncClients.forEach(client -> {
					client.initialize().subscribe(result -> {
						/* Client initialized successfully */ }, error -> {
							/* Log initialization error */ });
				});
			}
		}

	}

	/**
	 * Record class that implements {@link AutoCloseable} to ensure proper cleanup of MCP
	 * async clients.
	 *
	 * <p>
	 * This class is responsible for closing all MCP async clients when the application
	 * context is closed, preventing resource leaks.
	 */

	public record CloseableMcpAsyncClients(List<McpAsyncClient> clients) implements AutoCloseable {
		@Override
		public void close() {
			this.clients.forEach(McpAsyncClient::close);
		}
	}

}
