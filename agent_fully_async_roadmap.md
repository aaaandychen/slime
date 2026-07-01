# Agent Fully Async 开发 Roadmap

## 背景与目标

### 当前架构理解

SearchAgent-Zero 项目已有两层 async 抽象：

| 层级 | 位置 | 描述 | 状态 |
|------|------|------|------|
| **Agent Loop 内部** | `verl/experimental/agent_loop/tool_agent_loop.py` | 单样本内 asyncio 状态机，同 turn 内 tool calls 并行执行 | ✅ 已有 |
| **Worker 级别** | `verl/experimental/agent_loop/agent_loop.py:604` | 多样本间 `asyncio.create_task` + `asyncio.gather` 并行 | ✅ 已有 |
| **训练级别 (单步 Fully Async)** | `verl/experimental/fully_async_policy/` | Rollouter 和 Trainer 解耦为独立 Ray Actor，通过 MessageQueue 异步通信 | ✅ 已有 |
| **Agent 级别 Fully Async** | — | 单次 LLM 生成与 Tool 执行之间解耦（流式检测 + 提前调度） | ❌ 待开发 |

### 核心瓶颈

当前 `ToolAgentLoop` 的单步执行流程是**串行阻塞**的：

```
GENERATING（等待 LLM 完整输出）
    ↓ 全部 token 生成完才解析 tool calls
PROCESSING_TOOLS（等待所有 tool 执行完）
    ↓ 全部 tool 返回才进入下一轮
GENERATING（等待 LLM 完整输出）
    ↓
...
```

每个 turn 的总延迟 = **max(LLM 生成时间) + max(所有 tool 执行时间)**，无法重叠。

### "Agent Fully Async" 的定义

**目标**：将 Agent 单步内的 LLM 生成与 Tool 执行从串行改为流水线式并行，使得：

- LLM 生成到第一个 tool call 时，不等剩余 token 生成完，**立即调度 tool 执行**
- 第一个 tool 返回结果后，**不等同轮其他 tool**，立即开始下一轮 LLM 生成
- 多个 search query 之间的检索可以真正并行

```
时间线（当前）：
|-- LLM 生成(3个tool call) --|-- Tool1 --|-- Tool2 --|-- Tool3 --|-- LLM 生成 --|
总耗时 = T_gen + T_tool1 + T_tool2 + T_tool3 + T_gen2

时间线（目标）：
|-- LLM 边生成边检测 --|
  |-- Tool1(提前调度) --|
    |-- Tool2 --|
      |-- Tool3 --|
        |-- LLM 继续生成(流式消费tool结果) --|
总耗时 ≈ T_gen（与 tool 执行最大重叠）
```

---

## Phase 1: 深入理解现有代码（1-2天）

### 1.1 Agent Loop 核心流程

- [ ] **精读 `tool_agent_loop.py` 完整状态机**
  - `_handle_pending_state`：prompt 模板构建
  - `_handle_generating_state`：LLM 调用 → token 输出 → tool call 解析
  - `_handle_processing_tools_state`：tool 并行执行 → 摘要压缩 → prompt 拼接
  - `AgentData` 状态管理：`prompt_ids`, `response_mask`, `messages`, `response_ids`
  - 异常轨迹检测机制：`abnormal_trajectory_dic`

- [ ] **精读 `agent_loop.py` 基础设施**
  - `AsyncLLMServerManager.generate()`：LLM 调用的实际接口，理解 sticky session 和 load balancing
  - `AgentLoopWorker.generate_sequences()`：批量样本并行调度
  - `AgentLoopManager`：资源管理和 worker 生命周期

- [ ] **精读 `fully_async_policy/` 训练级异步**
  - `FullyAsyncRollouter._streaming_generation_main()`：连续样本生成循环
  - `FullyAsyncLLMServerManager.generate()`：partial rollout 恢复机制（理解 `while True` 循环内的分段生成）
  - `MessageQueue`：生产者-消费者模型（`asyncio.Condition`）

### 1.2 关键接口梳理

- [ ] 画出当前 ToolAgentLoop 的完整时序图
- [ ] 标注每个 `await` 点的阻塞范围
- [ ] 列出所有需要改为非阻塞的接口

---

## Phase 2: 流式 LLM 生成 + 早期 Tool Call 检测（3-5天）

### 2.1 Streaming Generator 抽象

**目标**：将 `server_manager.generate()` 从"一次性返回全部 token"改为"流式 yield token chunk"，同时保留向后兼容。

- [ ] **设计 `StreamingTokenOutput` 协议**
  ```python
  @dataclass
  class StreamingTokenChunk:
      token_ids: list[int]        # 增量 token
      log_probs: list[float] | None
      finish_reason: str | None   # "stop", "tool_call_detected", "length"
      extra_fields: dict

  # 新接口
  async def generate_stream(
      self, request_id, prompt_ids, sampling_params, ...
  ) -> AsyncIterator[StreamingTokenChunk]:
  ```

- [ ] **在 `AsyncLLMServerManager` 中实现 `generate_stream()`**
  - 基于现有 vLLM/SGLang 的 streaming API 封装
  - 保持 sticky session 和 load balancing 逻辑
  - 处理 preemption 和 partial rollout 场景

- [ ] **在 `FullyAsyncLLMServerManager` 中适配**
  - 继承并覆盖 `generate_stream()`，保持 `while True` 分段恢复逻辑
  - 确保 partial rollout resume 后 streaming 状态正确

### 2.2 Tool Call 增量解析器

**目标**：不等待 LLM 完整输出，在流式 token 到达时尽早检测到完整的 tool call。

- [ ] **实现 `IncrementalToolParser`**
  - 继承现有 `ToolParser`，添加增量解析能力
  - 每收到一个 chunk，尝试检测是否已有完整 JSON function call
  - 支持的格式：`gpt-oss`（function call 格式）、`tool_call` 标签格式等
  - 返回三种状态：
    - `INCOMPLETE`：继续接收 token
    - `TOOL_CALL_READY`：检测到完整 tool call，返回已解析的 `FunctionCall`
    - `TEXT_READY`：没有 tool call，纯文本生成完成

- [ ] **边界 case 处理**
  - JSON 跨 chunk 边界的情况（需要缓冲拼接）
  - 多个 tool calls 在同一轮生成中的增量检测
  - tool call 和普通文本混合的情况

### 2.3 验证测试

- [ ] **单元测试**：用 mock token stream 测试增量解析器
- [ ] **集成测试**：用真实模型测试流式生成 + 早期检测的延迟改善
- [ ] **基准对比**：记录当前模式 vs 流式模式的端到端延迟

---

## Phase 3: 生成与 Tool 执行的流水线化（5-7天）

### 3.1 AsyncAgentLoop 状态机重构

**目标**：将现有 `ToolAgentLoop` 的串行状态机改为流水线式。

- [ ] **设计 `StreamingAgentLoop` 新状态机**
  ```
  PENDING → STREAMING_GENERATING → (并发)
                                      ├─ 继续接收 token
                                      └─ 执行已检测到的 tool call
                                   → PROCESSING_TOOLS
                                   → STREAMING_GENERATING
                                   → ... → TERMINATED
  ```

- [ ] **核心实现：`_handle_streaming_generating_state()`**
  ```python
  async def _handle_streaming_generating_state(self, agent_data, sampling_params):
      tool_dispatch_tasks = []  # 已调度的 tool 执行任务
      accumulated_tokens = []
      
      async for chunk in self.server_manager.generate_stream(...):
          accumulated_tokens.extend(chunk.token_ids)
          # 增量解析
          status, tool_calls = self.incremental_parser.try_parse(accumulated_tokens)
          
          if status == ParseStatus.TOOL_CALL_READY:
              # 不等剩余 token，立即调度 tool 执行
              for tc in tool_calls:
                  task = asyncio.create_task(self._call_tool(tc, ...))
                  tool_dispatch_tasks.append(task)
          
          if chunk.finish_reason:
              break
      
      # 等待所有提前调度的 tool 执行完成
      if tool_dispatch_tasks:
          results = await asyncio.gather(*tool_dispatch_tasks)
          # 立即开始下一轮（不等 stream 完全结束）
      
      return next_state
  ```

- [ ] **异步状态管理增强**
  - `AgentData` 添加并发安全机制（`asyncio.Lock` 保护共享状态）
  - 支持"部分 tool 结果已返回，部分还在执行"的中间状态

### 3.2 提前调度优化

- [ ] **Tool 调度策略**
  - **立即调度**：检测到完整 tool call 立即 `create_task`，不等 stream 结束
  - **批量调度**：累积 N 个 tool call 或时间窗口到达后批量调度
  - **优先级调度**：基于 tool 类型设置优先级（search > 其他）

- [ ] **结果回调机制**
  - Tool 完成后通过 callback 更新 `agent_data.messages`
  - 支持 tool 结果流式回传（大文档分段注入 prompt）

### 3.3 验证测试

- [ ] **端到端延迟测试**：对比 ToolAgentLoop vs StreamingAgentLoop
- [ ] **正确性测试**：确保流水线化不改变 agent 行为（相同输入 → 相同输出）
- [ ] **吞吐量 benchmark**：测量在相同 GPU 资源下的 samples/second 提升

---

## Phase 4: 多分支并行搜索（3-5天）

### 4.1 Parallel Search Branches

**目标**：允许 Agent 在同一 turn 内启动多个独立搜索分支，并行探索不同方向。

- [ ] **设计 `BranchingSearchExecutor`**
  - 维护多个独立的搜索上下文（每个分支有自己的 `AgentData`）
  - 分支间共享已搜索的 query 去重集合
  - 分支合并策略：第一个返回满意结果即终止其他分支

- [ ] **分支管理**
  ```python
  class SearchBranch:
      agent_data: AgentData
      priority: int
      parent_branch_id: str | None
      
  class BranchingSearchExecutor:
      active_branches: dict[str, SearchBranch]
      max_concurrent_branches: int
      
      async def spawn_branch(self, query: str) -> str: ...
      async def merge_branch(self, branch_id: str) -> AgentData: ...
      async def cancel_branch(self, branch_id: str): ...
  ```

### 4.2 与流式生成的集成

- [ ] **IncrementalToolParser 支持分支检测**
  - 当模型输出多个独立搜索方向时，自动创建对应分支
  - 分支间的 LLM 生成共享 sticky session 以利用 prefix cache

---

## Phase 5: 与 Fully Async Policy 集成（3-5天）

### 5.1 Agent Async × Training Async

**目标**：确保 Agent 级别的流水线化与训练级别的 Rollouter/Trainer 解耦正确组合。

- [ ] **Partial Rollout 与 Streaming 的交互**
  - 当参数同步导致 partial rollout 中断时，streaming 状态如何恢复
  - 分段生成的 token 是否与 tool 执行状态一致
  - 设计中断安全的状态快照机制

- [ ] **MessageQueue 粒度调整**
  - 当前是"完整 agent trajectory"作为一条消息
  - Agent async 后可能改为"部分结果即可消费"的细粒度消息

### 5.2 配置系统

- [ ] **新增配置项**
  ```yaml
  actor_rollout_ref:
    rollout:
      multi_turn:
        # 流式生成
        enable_streaming_generation: true
        streaming_detection_strategy: "eager"  # eager | batch | adaptive
        
        # 提前调度
        enable_early_tool_dispatch: true
        max_concurrent_tool_executions: 8
        
        # 分支搜索
        enable_branching_search: false
        max_parallel_branches: 4
        branch_merge_strategy: "first_success"  # first_success | best_score | all
  ```

- [ ] **向后兼容**：默认全部关闭，不影响现有训练流程

---

## Phase 6: 性能评估与优化（3-5天）

### 6.1 Benchmark Suite

- [ ] **延迟指标**
  - Average Turn Latency（每轮 LLM 生成 + Tool 执行的总延迟）
  - Time To First Tool Call（TTFTC）
  - End-to-End Trajectory Latency

- [ ] **吞吐量指标**
  - Samples Per Second（SPS）
  - GPU Utilization Rate
  - Tool Execution Concurrency Ratio

- [ ] **质量指标**
  - 与同步模式的 reward score 对比
  - 异常轨迹率变化
  - Tool call 成功率

### 6.2 优化方向

- [ ] **Token Buffer 优化**
  - 增量解析器的 buffer 大小自适应
  - 避免不必要的 token 拷贝

- [ ] **调度策略调优**
  - 对比 eager/batch/adaptive 三种调度策略的性能差异
  - 基于模型类型（快/慢）自适应调整调度窗口

- [ ] **GPU Memory 优化**
  - Streaming 模式下 KV cache 管理
  - 多个并行分支的显存占用

---

## Phase 7: 文档与发布（1-2天）

- [ ] **编写使用文档**
  - `docs/agent_fully_async.md`：功能说明、配置指南、性能对比
  - 更新 README 的功能矩阵
- [ ] **编写示例脚本**
  - 新增 `run_xxx_fully_async_agent.sh` 启动脚本
  - 更新 example configs
- [ ] **代码清理**
  - 确保所有新增代码有完整的 type hints 和 docstrings
  - 跑通 pre-commit hooks

---

## 总结

| Phase | 内容 | 预计时间 | 依赖 |
|-------|------|----------|------|
| Phase 1 | 深入理解现有代码 | 1-2天 | 无 |
| Phase 2 | 流式生成 + 早期 Tool Call 检测 | 3-5天 | Phase 1 |
| Phase 3 | 生成与 Tool 执行流水线化 | 5-7天 | Phase 2 |
| Phase 4 | 多分支并行搜索 | 3-5天 | Phase 3 |
| Phase 5 | 与 Fully Async Policy 集成 | 3-5天 | Phase 3,4 |
| Phase 6 | 性能评估与优化 | 3-5天 | Phase 5 |
| Phase 7 | 文档与发布 | 1-2天 | Phase 6 |
| **总计** | | **19-31天** | |

### 核心改动文件

| 文件 | 改动类型 |
|------|----------|
| `verl/experimental/agent_loop/tool_agent_loop.py` | 重写状态机为流式 |
| `verl/experimental/agent_loop/agent_loop.py` | 新增 `generate_stream()` 接口 |
| `verl/experimental/agent_loop/tool_parser.py` | 新增 `IncrementalToolParser` |
| `verl/experimental/fully_async_policy/agent_loop/agent_loop.py` | 适配 streaming 到 partial rollout |
| `verl/experimental/agent_loop/agent_data.py` (如有) | 并发安全状态管理 |
| `verl/experimental/fully_async_policy/message_queue.py` | 粒度调整 |
| `verl/trainer/config/rollout/rollout.yaml` | 新增配置项 |

### 风险点

1. **vLLM/SGLang streaming API 兼容性**：不同推理后端的 streaming 行为可能不一致，需要抽象层
2. **Partial Rollout 状态一致性**：分段生成 + 提前 tool 调度的组合可能产生边界 bug
3. **并发正确性**：token 生成、tool 执行、状态更新三者并发时需要仔细设计锁粒度
4. **模型能力依赖**：流式 tool call 检测对模型输出格式有要求，某些小模型可能不稳定
