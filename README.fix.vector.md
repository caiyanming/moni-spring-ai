# Spring AI VectorStore 修复记录

## 问题背景

在启动 `moni-ai-agent` 时遇到以下错误：
```
Parameter 1 of method timeAwareMemoryManager in com.fufenxi.moni.ai.config.memory.MemoryServiceConfiguration 
required a bean of type 'org.springframework.ai.vectorstore.VectorStore' that could not be found.
```

## 根本原因分析

1. **核心问题**：Spring Boot 启动时找不到 VectorStore Bean
2. **期望解决方案**：使用 Spring AI 的 autoconfigure 机制自动创建 Qdrant VectorStore
3. **实际阻碍**：Qdrant autoconfigure 模块存在 API 兼容性问题

## 技术调查结果

### 1. 项目当前状态
- **Spring AI 版本**：`2.0.0-reactive-1` (本地构建的响应式版本)
- **Qdrant 客户端版本**：`io.qdrant:client:1.14.1`
- **Spring AI Alibaba 版本**：`2.0.0-reactive-1`

### 2. 模块构建状态
✅ **已成功构建的模块**：
- `spring-ai-commons:2.0.0-reactive-1`
- `spring-ai-model:2.0.0-reactive-1`
- `spring-ai-vector-store:2.0.0-reactive-1`
- `spring-ai-qdrant-store:2.0.0-reactive-1`
- `spring-ai-retry:2.0.0-reactive-1`
- `spring-ai-transformers:2.0.0-reactive-1`
- `spring-ai-autoconfigure-vector-store-qdrant:2.0.0-reactive-1` ✅ **已修复**

❌ **存在问题的模块**：
- `spring-ai-client-chat` - **测试代码编译错误**

### 3. Qdrant Autoconfigure API 兼容性问题

**问题文件**：`/auto-configurations/vector-stores/spring-ai-autoconfigure-vector-store-qdrant/src/main/java/org/springframework/ai/vectorstore/qdrant/autoconfigure/QdrantVectorStoreAutoConfiguration.java`

**具体错误**：
```java
// 第63行：API 不匹配
QdrantGrpcClient.Builder grpcClientBuilder = QdrantGrpcClient.newBuilder(
    connectionDetails.getHost(), connectionDetails.getPort(), properties.isUseTls()
);
// 错误：newBuilder 方法不接受参数

// 第67行：方法不存在
grpcClientBuilder.withApiKey(connectionDetails.getApiKey());
// 错误：withApiKey 方法不存在

// 第69行：构造器问题
return new QdrantClient(grpcClientBuilder.build());
// 错误：QdrantClient 是抽象类，无法直接实例化
```

### 4. ChatClient 模块问题

**问题**：测试代码中存在响应式 API 类型不匹配错误
**影响**：无法构建 `spring-ai-client-chat` 模块，导致应用启动时缺少 ChatClient Bean

## 尝试过的解决方案

### 方案1: 使用 SimpleVectorStore 作为备用 ❌
- **优点**：能让应用启动
- **缺点**：只是内存存储，不是真正的解决方案
- **状态**：已实现但不理想

### 方案2: 降级 Qdrant 客户端版本 ❌
- **想法**：找到与 Spring AI 兼容的旧版本
- **问题**：违背了保持技术栈先进性的原则

### 方案3: 修改应用代码使用 ChatModel 替代 ChatClient ❌
- **问题**：ChatClient 提供了更高级的 API，降级使用 ChatModel 增加了复杂性
- **结论**：不应该因为依赖问题而降级应用架构

## 推荐解决方案：修复 Spring AI 集成代码

### 为什么这是最优雅的解法

1. **从根本解决问题** - 不是绕过而是解决
2. **保持技术栈先进性** - 使用最新的 Qdrant 客户端 1.14.1
3. **符合开源精神** - 修复后可贡献回 Spring AI 社区
4. **学习价值** - 深入理解 Spring AI 和 Qdrant 的集成机制

### 需要修复的文件

1. **QdrantVectorStoreAutoConfiguration.java** (第63-69行)
   - 修复 QdrantGrpcClient 的创建方式
   - 更新 API Key 设置方法
   - 修复 QdrantClient 实例化

2. **ChatClient 测试代码** (可选)
   - 修复响应式类型匹配问题
   - 或者跳过测试进行构建

## ✅ 修复完成记录

### 修复日期：2025-01-12 16:30 - 17:30

**修复内容**：成功修复了 `QdrantVectorStoreAutoConfiguration.java` 中的 API 兼容性问题，以及测试代码中的响应式API问题

**修复前的错误代码**：
```java
// 错误的API调用
QdrantGrpcClient.Builder grpcClientBuilder = QdrantGrpcClient.newBuilder(
    connectionDetails.getHost(), connectionDetails.getPort(), properties.isUseTls()
);
grpcClientBuilder.withApiKey(connectionDetails.getApiKey());
return new QdrantClient(grpcClientBuilder.build());
```

**修复后的正确代码**：
```java
// 1. 创建 gRPC ManagedChannel
String hostPort = connectionDetails.getHost() + ":" + connectionDetails.getPort();
ManagedChannel channel;

if (properties.isUseTls()) {
    channel = Grpc.newChannelBuilder(hostPort, TlsChannelCredentials.create()).build();
} else {
    channel = Grpc.newChannelBuilder(hostPort, InsecureChannelCredentials.create()).build();
}

// 2. 创建 QdrantGrpcClient builder
QdrantGrpcClient.Builder grpcClientBuilder = QdrantGrpcClient.newBuilder()
        .channel(channel);

// 3. 设置 API key（如果提供）
if (connectionDetails.getApiKey() != null) {
    grpcClientBuilder.callCredentials(new ApiKeyCredentials(connectionDetails.getApiKey()));
}

return grpcClientBuilder.build();
```

### 测试代码修复

**修复前**：使用了阻塞式 `.block()` 调用和旧的 Qdrant API
**修复后**：使用 `StepVerifier` 进行纯响应式测试，使用正确的 Qdrant 1.14.1 响应式API

**关键发现**：
1. **响应式版本的VectorStore接口**：返回 `Flux<Document>` 而非 `List<Document>`
2. **Qdrant 1.14.1 API架构**：
   - 实际：`QdrantGrpcClient.newBuilder()` 无参数
   - 期望：`QdrantGrpcClient.newBuilder(host, port, useTls)`
3. **认证方式**：
   - 实际：使用 `ApiKeyCredentials` 类和 `callCredentials()` 方法
   - 期望：使用 `withApiKey()` 方法
4. **ManagedChannel需要手动创建**：
   - 必须使用 `Grpc.newChannelBuilder()` 创建连接
   - 需要根据 `useTls` 参数选择 `TlsChannelCredentials` 或 `InsecureChannelCredentials`
5. **响应式API方法名**：
   - `listCollections()` → `Flux<CollectionDescription>`
   - `deleteCollection(String)` → `Mono<CollectionOperationResponse>`
   - `createCollection(CreateCollection)` → `Mono<CollectionOperationResponse>`

### 新增依赖imports：
```java
import io.grpc.Grpc;
import io.grpc.InsecureChannelCredentials;
import io.grpc.ManagedChannel;
import io.grpc.TlsChannelCredentials;
import io.qdrant.client.ApiKeyCredentials;
import reactor.test.StepVerifier;
```

### 测试代码修复策略：
1. **纯响应式测试**：使用 `StepVerifier` 替代所有 `.block()` 调用
2. **正确的API调用**：使用 Qdrant 1.14.1 的真实响应式方法
3. **完整的集合管理**：修复 `@BeforeAll` 中的集合创建逻辑

## 下一步计划

### 立即任务
1. ✅ **研究 Qdrant 1.14.1 正确 API** 
   - ✅ 查看官方文档或源码
   - ✅ 了解正确的客户端创建方式

2. ✅ **修复 QdrantVectorStoreAutoConfiguration.java**
   - ✅ 更新 QdrantGrpcClient.newBuilder() 调用
   - ✅ 修复 API Key 设置
   - ✅ 修复 QdrantClient 实例化

3. ✅ **修复测试代码**
   - ✅ 修复响应式API的正确使用
   - ✅ 使用StepVerifier进行纯响应式测试
   - ✅ 修复旧的Qdrant API调用

4. ✅ **重新构建和安装**
   - ✅ 重新构建 autoconfigure 模块
   - ✅ 成功安装到本地Maven仓库

5. **在 moni-ai-agent 中测试**
   - 启用 autoconfigure 依赖
   - 验证 VectorStore Bean 正确创建
   - 测试应用完整启动

### 后续任务
1. **ChatClient 问题处理**
   - 修复测试代码或跳过测试构建
   - 确保 ChatClient Bean 可用

2. **集成测试**
   - 完整测试应用启动
   - 验证 VectorStore 功能正常

3. **文档更新**
   - 更新 CLAUDE.md 中的版本信息
   - 记录修复过程和经验

## 技术细节备忘

### 当前 build.gradle 配置
```gradle
// Spring AI 版本 - 使用本地构建的响应式版本  
set('springAIVersion', '2.0.0-reactive-1')
// Spring AI Alibaba 版本 - 使用本地构建的响应式版本
set('springAIAlibabaVersion', '2.0.0-reactive-1')

// Qdrant向量数据库依赖
implementation "org.springframework.ai:spring-ai-qdrant-store:${springAIVersion}"
implementation "org.springframework.ai:spring-ai-autoconfigure-vector-store-qdrant:${springAIVersion}" // 现在可以启用！
implementation 'io.qdrant:client:1.14.1'
```

### application.yml 中的 Qdrant 配置
```yaml
spring:
  ai:
    vectorstore:
      qdrant:
        host: ${QDRANT_HOST:localhost}
        port: ${QDRANT_PORT:6334}
        api-key: ${QDRANT_API_KEY:your-api-key}
        collection-name: ${QDRANT_COLLECTION:moni}
        use-tls: ${QDRANT_USE_TLS:false}
        initialize-schema: true
```

## 关键学习点

1. **Spring AI 模块化设计**：core、model、vectorstore、autoconfigure 分层清晰
2. **API 兼容性重要性**：小版本升级也可能带来不兼容变更
3. **响应式API的全栈设计**：从VectorStore接口到测试代码都需要保持响应式
4. **测试驱动修复**：通过测试代码了解正确 API 用法
5. **渐进式解决**：先解决核心问题，再处理外围问题
6. **深入源码分析**：查看实际的客户端实现是解决问题的关键
7. **纯响应式测试**：使用StepVerifier替代.block()调用的最佳实践

---

**最后更新**：2025-01-12 17:30  
**当前状态**：✅ Qdrant API 兼容性问题已完全修复，模块已成功安装到本地Maven仓库  
**下次继续**：在moni-ai-agent中启用autoconfigure依赖并测试应用启动