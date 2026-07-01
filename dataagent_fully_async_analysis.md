# DataAgent × Slime Fully Async RL 训练 —— 问题分析

## 背景

### Slime

Slime 是一个基于 Ray 的分布式 LLM RL 训练框架。核心流程：

```
Rollout（SGLang 推理）→ Reward（打分）→ Training（PPO/GRPO）→ 权重同步 → 下一轮
```

支持三种训练模式：

| 模式 | 入口 | 特点 |
|------|------|------|
| 同步 | `sglang_rollout.generate_rollout` | 生成完一批 → 训练 → 下一批 |
| 异步 | `train_async.py` | 训练和下一轮生成并行（Ray future 预取），权重同步前等当前 rollout 完成 |
| Fully Async | `train_async.py` + `fully_async_rollout.py` | 后台 worker 持续生成，训练从 output_queue 取，权重同步时通过 `engine.pause_generation()` 中断在飞的 SGLang 请求 |

Fully async 的关键机制：
- `UpdateWeightFromDistributed.update_weights()` → `engine.pause_generation()` → 中断所有在飞的 SGLang HTTP 请求 → `engine.flush_cache()` → 广播新权重 → `engine.continue_generation()`
- 被中断的 `generate()` 返回 `finish_reason="abort"`，sample 标记 ABORTED → worker callback 将其 requeue 到 `data_buffer`
- 下次 rollout 取出 ABORTED sample → `_prepare_prompt_ids` 返回完整 `sample.tokens`（含旧权重生成的部分）→ SGLang prefix caching 跳过已生成部分 → 增量生成新 token

### DataAgent

DataAgent 是企业级智能数据分析 Agent（Spring Boot + Vue）。核心执行流是一个 StateGraph：

```
Intent 识别 → Evidence 召回(RAG) → Schema 查询 → TableRelation → Feasibility 评估
→ Planner 生成执行计划 → [循环]:
    SQL_Generate → Semantic 校验 → SQL_Execute → PlanExecutor
    → Python_Generate → Python_Execute(Docker) → Python_Analyze → PlanExecutor
→ Report_Generate → END
```

工具层与编排层干净分离。所有工具服务（SqlExecutor、CodePoolExecutorService、AgentVectorStoreService）对 StateGraph 零依赖，可独立调用。

### 目标

用 DataAgent 的工具执行能力（SQL 执行器、Python 执行器、Schema 查询、RAG 检索）作为 Slime RL 训练的 rollout 环境。让 LLM 通过 RL 学会数据分析的多步规划与工具调用，而非依赖 DataAgent 固定的 StateGraph 控制流。

## 发现

### 1. 所有现有多轮 Agent 示例都禁用了 partial rollout

- [tau-bench](examples/tau-bench/generate_with_tau.py#L120): `assert not args.partial_rollout`
- [strands](examples/strands_sglang/generate_with_strands.py#L44): `assert not args.partial_rollout`
- [coding_agent_rl](examples/coding_agent_rl/generate.py): 不使用 `generate()`，通过适配器代理 SGLang 通信

### 2. `generate()` 是单轮 API

`sglang_rollout.py:161-163`：
```python
assert sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
```

第一次 `generate()` 调用后，`_apply_meta_info` 将 status 设为 COMPLETED（finish_reason="stop"）或 TRUNCATED（"length"）。第二次调用 `generate()` 直接 assert 失败。

这意味着自定义多轮 agent 函数**不能**通过多次调用框架的 `generate()` 来实现 `LLM → Tool → LLM → Tool` 循环。现有示例全部绕开 `generate()`，自己封装 SGLang HTTP 通信和 token 管理。

### 3. Fully async 的中断式 rollout 实际存在

之前误认为 `fully_async_rollout.py` + `train_async.py` 没有中断机制。实际情况：

`UpdateWeightFromDistributed.update_weights()` 调用 `engine.pause_generation()` → 中断所有在飞的 SGLang 请求 → `flush_cache()` → 广播权重 → `continue_generation()`。

后台 worker 的 `generate()` HTTP 请求被中断，返回 `finish_reason="abort"`，sample 被 requeue。**中断式 rollout 确实存在，但中断只发生在 `generate()` 内部（LLM 推理），不发生在 tool 执行期间。**

## 问题定位

经过对话的逐步排查，问题收敛到一点：

**`generate()` 被设计为单轮 API。多轮 agent 必须绕开它自己做 SGLang HTTP 通信和 token 管理。但绕开后，框架的 partial rollout 恢复路径（`_prepare_prompt_ids` 返回完整 tokens → SGLang 续写）仍然只在 token 层面正确工作——自定义函数在控制流层面没有任何恢复支持。**

具体：

1. 自定义函数在 ABORTED sample 上被重新调用时，拿到的是一个扁平的 token 序列（含之前所有 turn 的模型输出和工具结果）和一个 `ABORTED` 状态
2. 没有 turn 边界的标记——不知道 `sample.tokens` 中哪些是 turn1 的 SQL、哪些是 turn1 的工具结果、哪些是被截断的 turn2 partial tokens
3. 没有"哪些工具已执行"的元数据——不知道该从 agent loop 的第几个 turn 继续
4. 没有标准的恢复入口——框架的 `generate()` 是单轮 API，无法作为多轮循环中的"继续生成"原语

## 当前讨论点

**这个问题应该在哪里解决？**

- 框架层？让 `generate()` 支持多次调用、让 Sample 携带 segment 边界、让恢复路径感知工具结果？
- 自定义函数层？自己维护状态元数据、自己做断点恢复？
- DataAgent 工具封装层？工具 SDK 提供幂等性保证和状态序列化？

**下一步**：确定问题的最佳解决位置，然后设计具体方案。

## 历史分析：Fully Async 特性矩阵

以下矩阵来自 [agent_fully_async_design.md](agent_fully_async_design.md) 和 [fully_async_code_guide.md](fully_async_code_guide.md)，结合 DataAgent 执行流做了映射：

| ID | 特性 | Slime 当前状态 | DataAgent 场景的意义 |
|----|------|-------------|-------------------|
| T5 | Staleness 管理 | 框架有 `reset_staleness`，但无阈值控制 | DataAgent 轨迹 10-60s，跨版本是常态 |
| A1 | 流式生成 | `sglang_streaming_rollout.py` 已有 SSE 消费 | 可复用于流式消费多 turn 输出 |
| A2 | 增量 tool 检测 | 缺。现有 parser 只支持 XML/JSON | 需要代码块 fence 检测（\`\`\`sql / \`\`\`python） |
| A3 | tool 提前调度 | 缺 | DataAgent 价值最大：同 turn 内多个独立 SQL 可并行 |
| A4 | 异步结果注入 | 缺 | 大 SQL 结果可分段注入上下文 |
| C1 | Tool token 保护 | `mask_offpolicy` 全部归零，但不区分 model/tool | 工具结果是环境观察，不可丢弃 |
| C2 | 安全中断点 | 天然存在（abort 只在 `generate()` 内部） | 工具执行不受 abort 影响 |
| C3 | 轨迹 staleness | 无 per-turn 追踪 | 跨版本轨迹中各 segment 版本不明 |
| C4 | Agent 感知流控 | 无 | 需要感知当前轨迹状态决定暂停策略 |

## 关键代码引用

- [`generate_and_rm`](slime/rollout/sglang_rollout.py#L223-L286) — 单样本生成+奖励入口，调用自定义函数
- [`generate`](slime/rollout/sglang_rollout.py#L153-L220) — 单轮 SGLang 生成，有 status assert
- [`_prepare_prompt_ids`](slime/rollout/sglang_rollout.py#L43-L62) — 恢复时返回完整 `sample.tokens`
- [`fully_async_rollout._loop`](slime/rollout/fully_async_rollout.py#L147-L196) — 后台 worker 持续生成
- [`fully_async_rollout._make_done_cb`](slime/rollout/fully_async_rollout.py#L198-L220) — ABORTED → requeue
- [`update_weights`](slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L102-L133) — pause/flush/broadcast/continue
- [`train_async.py`](train_async.py#L30-L76) — 异步训练主循环
- [`Sample.append_response_tokens`](slime/utils/types.py#L253-L314) — token 追加接口，支持 `trainable` 参数
