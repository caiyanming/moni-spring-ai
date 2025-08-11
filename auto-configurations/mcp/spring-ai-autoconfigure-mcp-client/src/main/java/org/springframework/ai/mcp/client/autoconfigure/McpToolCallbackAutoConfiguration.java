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

import io.modelcontextprotocol.client.McpAsyncClient;

import org.springframework.ai.mcp.AsyncMcpToolCallbackProvider;
import org.springframework.ai.mcp.client.autoconfigure.properties.McpClientCommonProperties;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.AllNestedConditions;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Conditional;

/**
 */
@AutoConfiguration(after = { McpClientAutoConfiguration.class })
@EnableConfigurationProperties(McpClientCommonProperties.class)
@Conditional(McpToolCallbackAutoConfiguration.McpToolCallbackAutoConfigurationCondition.class)
public class McpToolCallbackAutoConfiguration {

	/**
	 * Creates async tool callbacks for all configured MCP clients.
	 *
	 * <p>
	 * These callbacks enable integration with Spring AI's tool execution framework,
	 * allowing MCP tools to be used as part of AI interactions in a reactive manner.
	 * @param mcpClientsProvider provider of MCP async clients
	 * @return async tool callback provider for MCP integration
	 */

	@Bean
	@ConditionalOnProperty(prefix = McpClientCommonProperties.CONFIG_PREFIX, name = "enabled", havingValue = "true",
			matchIfMissing = true)
	public AsyncMcpToolCallbackProvider mcpToolCallbacks(ObjectProvider<List<McpAsyncClient>> mcpClientsProvider) {
		List<McpAsyncClient> mcpClients = mcpClientsProvider.stream().flatMap(List::stream).toList();
		return new AsyncMcpToolCallbackProvider(mcpClients);
	}

	public static class McpToolCallbackAutoConfigurationCondition extends AllNestedConditions {

		public McpToolCallbackAutoConfigurationCondition() {
			super(ConfigurationPhase.PARSE_CONFIGURATION);
		}

		@ConditionalOnProperty(prefix = McpClientCommonProperties.CONFIG_PREFIX, name = "enabled", havingValue = "true",
				matchIfMissing = true)
		static class McpAutoConfigEnabled {

		}

		@ConditionalOnProperty(prefix = McpClientCommonProperties.CONFIG_PREFIX + ".toolcallback", name = "enabled",
				havingValue = "true", matchIfMissing = true)
		static class ToolCallbackProviderEnabled {

		}

	}

}
