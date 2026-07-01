# Fully Async Agentic RL — 特性分解与实现分析

> 从第一性原理出发，定义 Fully Async Agentic RL 需要什么能力，再看 verl 和 slime 各自怎么实现、缺什么。

---

## 1. 特性矩阵

Fully Async Agentic RL = 训练级 Async + Agent 级 Async + 两者的组合。三层共 15 个独立特性：

### Layer 1：训练级 Async（Rollout 和 Train 解耦）

| ID | 特性 | 描述 |
|----|------|------|
| T1 | 独立进程 | Rollout 和 Train 跑在不同 GPU 上，各自独立循环 |
| T2 | 异步样本传递 | 生成完的样本不阻塞训练，训练不阻塞生成 |
| T3 | 权重同步 | 训练后的新权重能推送到正在生成中的推理引擎 |
| T4 | 持续生成 | 不按 batch 启停，后台持续跑 |
| T5 | Staleness 管理 | 追踪多少样本是用旧权重生成的，超阈值暂停生成 |
| T6 | 流控 / Backpressure | 队列满或 staleness 超 → 暂停生成；训练追上 → 恢复 |

### Layer 2：Agent 级 Async（单条轨迹内部异步）

| ID | 特性 | 描述 |
|----|------|------|
| A1 | 流式生成 | Token 级别消费 LLM 输出，不等完整响应 |
| A2 | 增量 tool 检测 | 从部分 token 流中提前检测到 tool call |
| A3 | tool 提前调度 | 检测到 tool call 立刻执行，不等 LLM 生成结束 |
| A4 | 异步结果注入 | tool 结果回来就注入上下文，不等其他 tool |
| A5 | token 计账 | model token（可训）和 tool token（不可训）正确标记 |

### Layer 3：组合问题（训练 async × Agent async）

| ID | 特性 | 描述 |
|----|------|------|
| C1 | Partial rollout 跨 tool 边界 | 权重同步中断轨迹时，之前的 tool 结果不能回滚，只能从 model token 恢复 |
| C2 | 安全中断点 | tool 执行中不能中断（有副作用），只能在纯 LLM 生成或 turn 间隙中断 |
| C3 | 轨迹级 staleness | 一条轨迹可能跨多个权重版本（第 1 轮用 v1 + tool，第 3 轮用 v2 生成）|
| C4 | Agent 感知的流控 | 暂停信号要考虑当前轨迹是否在 tool 执行中，不能暴力打断 |

---

## 2. Verl 实现分析

### Layer 1：完整实现

```
T1 ✅ FullyAsyncRollouter (Ray Actor) + FullyAsyncTrainer (Ray Actor)
      各自独立 GPU（4+4），各自 while True 循环，互为守护进程

T2 ✅ MessageQueue (Ray Actor)
      asyncio.Condition 生产者-消费者，有界队列 (deque(maxlen=...))
      MessageQueueClient 封装 Ray future → asyncio

T3 ✅ CheckpointEngineManager.update_weights()
      Trainer 每 trigger_parameter_sync_step 步推送权重到 vLLM replicas
      推送后调用 rollouter.reset_staleness() 恢复生成

T4 ✅ _processor_worker() while True 循环
      _feed_samples() 从 dataloader 逐条喂入 → asyncio.Queue → _processor_worker()
      逐条消费，max_concurrent_samples 限制并发

T5 ✅ staleness_samples 计数器
      _should_pause_generation() 检查两个条件：
        - queue_size >= max_queue_size（队列满）
        - staleness_samples >= max_required_samples（旧权重样本过多）
      max_required_samples = ppo_mini_batch_size × (staleness_threshold + 1) × trigger_parameter_sync_step

T6 ✅ _resume_event (asyncio.Event)
      pause 时 clear()，trainer 同步权重后 reset_staleness() → set() 唤醒
      双层流控：Rollouter 级 pause/resume + _processor_worker 级并发数限制
```

### Layer 2：待开发

```
A1 ❌ 缺 AsyncLLMServerManager.generate_stream()
      当前 generate() 一次性返回 TokenOutput，vLLM 有 streaming API 但未封装

A2 ❌ 缺 ToolParser.try_parse_partial()
      3 个 parser (Hermes/GptOss/Qwen3XML) 都是 decode(full_text) → regex
      需要增量接口：喂入累积文本 → 返回 INCOMPLETE / TOOL_CALL_READY / TEXT_DONE

A3 ❌ 缺 StreamingToolAgentLoop
      当前 _handle_generating_state 是"等完整输出 → 解析 tool call → 下一状态"
      需要改为"流式消费 chunk → 检测到 tool call 立刻 asyncio.create_task 调度"

A4 ⚠️ _handle_processing_tools_state 已有 asyncio.gather 并发执行 tool
      但缺"tool 结果异步回调注入到正在生成中的 LLM 流"的能力

A5 ✅ response_mask (1=model, 0=tool)
      AgentData 封装所有可变状态，ToolAgentLoop 管理 prompt_ids/response_ids/mask
```

**Layer 2 增量：~500 行**（`generate_stream()` 60 + `try_parse_partial()` 200 + `StreamingToolAgentLoop` 200 + config 5）

### Layer 3：部分已有，部分需补

```
C1 ⚠️ 巧妙之处：当前架构下 tool token 天然受保护
      ToolAgentLoop 在 _handle_processing_tools_state 中把 tool response token
      拼入 agent_data.prompt_ids（而非 response_ids）。FullyAsyncLLMServerManager
      的 while True 循环从 prompt_ids + final_output.token_ids 恢复生成，
      tool token 在 prompt_ids 中，不会被重新生成。
      
      但 Agent async 后这个假设失效：tool 在 generate() 内部就被 dispatch 了，
      final_output 中混入了 tool 执行期间的 model token，恢复边界不再清晰。
      
      需要补：tool 已 dispatch 标记 + tool token 不可回滚保护。

C2 ❌ _should_pause_generation() 在 Rollouter 层面（样本之间），不感知轨迹内部
      需要补：critical_section 信号（tool 执行中不允许中断）

C3 ⚠️ staleness_samples 计数器在 Rollouter 层面
      每个样本有 global_steps/min_global_steps/max_global_steps 元数据
      但轨迹内部跨版本没有精细追踪
      需要补：per-turn 版本号

C4 ❌ 流控只看队列大小和 staleness 总数，不看轨迹状态
      需要补：pause 信号发出后等待当前 tool 执行完成再真正暂停
```

**Layer 3 增量：~160 行**

### Verl 总计：~660 行

---

## 3. Slime 实现分析

### 架构差异

Slime 和 Verl 对"训练级 async"的设计理念不同：

```
Verl：两个对等独立 Actor，通过消息队列协商速度
  FullyAsyncRollouter ←── MessageQueue ──→ FullyAsyncTrainer
  各自 while True，互为守护，谁快了就等对方

Slime：训练为主循环，后台生成线程/进程辅助
  train_async.py 主循环
    ├── rollout_manager.generate.remote()  (Ray Actor, 预取)
    └── actor_model.async_train()         (本地 GPU)
  fully_async_rollout.py 更进一步：
    AsyncRolloutWorker 后台线程持续生成 → output_queue → 训练消费
```

### Layer 1

```
T1 ⚠️ 两种模式：
      train_async.py：Ray placement group 分离 GPU，但控制流是训练驱动的预取
        训练循环启动 rollout，等结果，训练，同时启动下一轮 rollout
        关键：训练主导节奏，不是两个对等循环
      fully_async_rollout.py：更进一步，后台线程持续生成，训练消费已完成的组
      
T2 ⚠️ 标准 async：Ray future 预取 (rollout_data_next_future)
      fully_async：线程安全 output_queue.Queue(maxsize=1000)
      没有独立的跨进程消息队列概念

T3 ✅ UpdateWeightFromDistributed / UpdateWeightFromDisk / UpdateWeightFromTensor
      三种模式，比 verl 更丰富（支持 delta 权重同步）
      train_async.py L67：训练先等当前生成完成，再 sync 权重，再启动新生成

T4 ✅ fully_async_rollout.AsyncRolloutWorker
      后台线程 + asyncio 循环，持续从 data_buffer 取样本 → generate_and_rm_group() → output_queue

T5 ❌ 没有 staleness 追踪
      训练循环每 update_weights_interval 步同步权重前，先 ray.get 等生成完成
      fully_async 的 ABORTED 样本 requeue 到 data_buffer，但不计数
      没有 staleness_threshold 概念

T6 ⚠️ output_queue.Queue(maxsize=1000) 提供基本背压
      但 pause/resume 是训练循环驱动的（等 ray.get 完成），不是队列压力驱动的
```

### Layer 2

```
A1 ✅ sglang_streaming_rollout.py:generate_streaming()
      逐 SSE chunk 消费，每个 chunk 立刻写回 sample.tokens
      比 verl 成熟：已经处理了 base_tokens 快照 + 增量恢复

A2 ❌ 缺增量 tool call 检测
      streaming 只做 SSE 消费，不解析 tool call
      但基础好：chunk 已经在逐块到达

A3 ❌ 缺提前调度逻辑

A4 ⚠️ search-r1 generate() 是串行调 search API + 拼字符串
      并发 tool 执行需要新写

A5 ✅ sample.append_response_tokens(trainable=True/False)
      search-r1 示例：tool 结果标记 loss_mask=0
```

### Layer 3

```
C1 ❌ --partial-rollout 把 ABORTED 样本 requeue 到 data_buffer
      下次 generate() 被调用时，样本上有部分 token
      但 generate() 不知道哪些是 model token（可重新生成）、哪些是 tool token（不能回滚）
      search-r1 的 generate() 用字符串累积 response，没有"从中间恢复"的概念
      这是 slime 最大的坑

C2 ❌ 没有安全中断点概念
      fully_async worker 的 abort 是 SGLang 层面的 (GenerateState.aborted)
      不感知 agent 内部状态

C3 ❌ 没有轨迹级 staleness 追踪

C4 ❌ 流控只看 output_queue 大小
```

---

## 4. 各自需要补什么

| ID | 特性 | Verl 增量 | Slime 增量 |
|----|------|----------|-----------|
| T5 | Staleness 管理 | 0（已有） | ~80 行 |
| T6 | 流控增强 | 0（已有） | ~60 行 |
| A1 | 流式生成 | ~60 行 | 0（已有） |
| A2 | 增量 tool 检测 | ~200 行 | ~200 行 |
| A3 | tool 提前调度 | ~100 行 | ~100 行 |
| A4 | 异步结果注入 | ~80 行 | ~80 行 |
| A5 | token 计账 | 0（已有） | 0（已有） |
| C1 | Tool token 保护 | ~50 行 | ~80 行 |
| C2 | 安全中断点 | ~50 行 | ~60 行 |
| C3 | 轨迹 staleness | ~30 行 | ~60 行 |
| C4 | Agent 感知流控 | ~30 行 | ~60 行 |
| **合计** | | **~600 行** | **~780 行** |

### 量差不多，但结构不同

```
Verl（~600 行）：
  ├── 改框架内部 4 个文件（AsyncLLMServerManager, ToolParser, FullyAsyncLLMServerManager, 新建 StreamingToolAgentLoop）
  ├── 不改外层 Rollouter/Trainer/MessageQueue
  └── Agent Loop 通过 @register + Hydra 配置接入

Slime（~780 行）：
  ├── 多 ~180 行在 Layer 1（T5/T6 staleness）和 Layer 3（C1-C4 组合逻辑）
  ├── 少 ~60 行在 Layer 2（A1 streaming 已有）
  ├── 核心逻辑写在一个自定义 generate() 函数里
  ├── 框架通过 --custom-generate-function-path 接入
  └── C1（tool token 保护）是最大风险点：
      需要 generate() 支持"从已有部分 token 恢复，跳过 tool token，只重新生成 model token"
```

### 关键差异

| | Verl | Slime |
|--|------|-------|
| 改动位置 | 框架内部（4 文件） | 1 个自定义 generate() 函数 |
| 框架改动量 | ~600 行 | ~200 行（T5/T6 框架补缺） |
| Agent 逻辑 | ~400 行（StreamingToolAgentLoop） | ~580 行（generate() 函数内） |
| 最大风险 | ToolParser 增量解析正确性 | C1: partial rollout 恢复 + agent 状态重建 |
| 接入方式 | 改 config 一行 | 改启动参数一行 |
