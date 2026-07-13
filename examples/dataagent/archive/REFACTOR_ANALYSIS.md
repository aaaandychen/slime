# DataAgent 黑盒 RL 复用 Slime 接口 —— 改造分析报告

## 1. 当前状态

### 1.1 文件职责

| 文件 | 行数 | 职责 | 是否可以删除 |
|------|------|------|-------------|
| `custom_generate.py` | 139 | slime `--custom-generate-function-path` 入口 | **重构** |
| `slime_api.py` | 264 | OpenAI 兼容 `/v1/chat/completions` 代理 → SGLang | **可删除** |
| `reward_func.py` | 133 | 多维度评分函数 | 保留 |
| `run_qwen3_14B_fully_async.sh` | 318 | 启动脚本 | 微调 |
| `test_abort_resume.py` | 323 | 端到端 abort/resume 测试 | 微调 |
| `check_output.py` | 63 | 独立检验脚本 | 微调 |
| `queries.jsonl` | 100 | 训练数据集 | 保留 |

### 1.2 当前架构

```
custom_generate.generate()
│
├── slime_api.ensure_server(args)       ← 手动启动 aiohttp 服务器
│
├── _sample_map[thread_id] = sample     ← 手动注册 Sample
│
├── _query_dataagent(query, thread_id)  ← SSE 调用 DataAgent
│     │
│     └── DataAgent (Java) 内部 8~12 轮 LLM 调用
│           │
│           └── POST /v1/chat/completions  ← slime_api 手动实现的端点
│                 │
│                 ├── 手动 chat template → tokenize
│                 ├── 手动 post(SGLang /generate)
│                 ├── 手动提取 token/logprob
│                 ├── 手动 sample.append_response_tokens()
│                 └── 手动 abort → 等待 notify_resume() → 重试
│
├── score(nodes)                        ← 手动调用 reward 函数
│
└── return sample                       ← 返回单个 Sample
```

### 1.3 核心问题：`slime_api.py` 重复实现了 `OpenAIAdapter`

`slime_api.py` 做的事情 = slime 已有的 `slime.agent.adapters.openai.OpenAIAdapter` 的子集：

| 功能 | dataagent `slime_api.py` | slime `OpenAIAdapter` |
|------|--------------------------|----------------------|
| `/v1/chat/completions` 路由 | 手动 aiohttp 注册 | `_register_routes` 自动注册 |
| Chat template → tokenize | 手动调用 | `_render_token_ids()` |
| SGLang `/generate` 代理 | 手动 `post()` | `call_sglang_generate()` |
| Token/logprob 捕获 | 手动提取 | `TurnRecord` 自动封装 |
| Session 生命周期 | 手动 dict (`_sample_map`) | `open_session` / `finish_session` / `drop_session` |
| 多轮对话管理 | ❌ 所有 token 塞进一个 Sample | `TrajectoryManager` 树结构 + 自动 fork/merge |
| loss_mask | 手动设置 | 自动区分 prompt (0) / output (1) |
| 多段轨迹 fan-out | ❌ 不支持 | DataAgent 线性对话 → 1 个 Sample。分支场景自动 fan-out |
| aiohttp 启动 | 手动 thread + event loop | `run_app_in_thread` |

## 2. 推荐架构

### 2.1 整体设计

```
custom_generate.generate()
│
├── DataAgentAdapter.ensure(args)       ← 复用 run_app_in_thread 启动适配器
│
├── adapter.open_session(thread_id,     ← 复用 BaseAdapter 标准 session 生命周期
│       sampling_defaults=...)
│
├── _query_dataagent(query, thread_id)  ← SSE 调用 DataAgent（保持不变）
│     │
│     └── DataAgent (Java) 内部 LLM 调用
│           │
│           └── POST /v1/chat/completions  ← OpenAIAdapter 自动处理
│                 │
│                 ├── _translate() → chat messages
│                 ├── _render_token_ids() → prompt_ids
│                 ├── call_sglang_generate_with_retry() → TurnRecord
│                 │     └── abort → 保存部分 token → 等待 resume → 续推
│                 ├── parse_model_output() → 工具调用/思维链
│                 ├── TrajectoryManager.record_turn() → 自动记录
│                 └── _respond() → SSE 流式返回
│
├── samples = adapter.finish_session(   ← 自动生成带正确 loss_mask 的 1 个 Sample（自动拼接所有 turn，首轮 prompt 剔除）
│       thread_id, reward=score(...))
│
└── return samples                      ← 返回 list[Sample]（DataAgent 线性对话 → 1 个 Sample，与当前一致）
```

### 2.2 组件清单

| 组件 | 新增/修改 | 预计行数 | 说明 |
|------|----------|---------|------|
| `adapter.py` | **新增** | ~60 行 | `DataAgentAdapter(OpenAIAdapter)` 子类 |
| `custom_generate.py` | **重构** | ~100 行 | 改用 adapter session 生命周期 |
| `slime_api.py` | **删除** | 0 行 | 功能全部由 adapter 接管 |
| `reward_func.py` | **保留** | 不变 | 纯评分函数 |

## 3. 详细改造方案

### 3.1 新增 `adapter.py` —— DataAgentAdapter

这是核心改造：一个 ~60 行的 `OpenAIAdapter` 子类，解决三个问题：

1. **Session ID 路由**：DataAgent 通过 `X-DataAgent-Thread-Id` Header 传递会话标识
2. **Abort/Resume 透明化**：SGLang 权重同步时，对 DataAgent 表现为"慢但不断"的 LLM
3. **工具调用解析**：DataAgent 内部使用 tool calling，需要正确的 tool_parser

```python
"""DataAgent adapter — OpenAI-compatible endpoint with abort-transparent retry.

Replaces the hand-rolled slime_api.py with a thin OpenAIAdapter subclass.
DataAgent's X-DataAgent-Thread-Id header maps to session IDs; the adapter's
TrajectoryManager records every turn automatically.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from slime.agent.adapters.common import (
    BaseAdapter,
    TurnRecord,
    call_sglang_generate,
)
from slime.agent.adapters.openai import OpenAIAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)


class DataAgentAdapter(OpenAIAdapter):
    """OpenAIAdapter subclass that reads session IDs from DataAgent headers."""

    def _session_id(self, request, body):
        # DataAgent passes thread ID via custom header.
        tid = request.headers.get("X-DataAgent-Thread-Id", "")
        if tid:
            return tid
        # Fall back to standard OpenAI mechanisms (metadata.session_id / user).
        return super()._session_id(request, body)


class AdapterService(metaclass=SingletonMeta):
    """Per-process singleton: tokenizer + adapter + aiohttp server thread.

    Mirrors coding_agent_rl's _AdapterService pattern.
    """

    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"

        tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        fork_threshold = (
            int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        )

        self.adapter = DataAgentAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=tool_parser,
            reasoning_parser=reasoning_parser,
            fork_threshold_tokens=fork_threshold,
        )

        host = getattr(args, "slime_api_host", "0.0.0.0")
        port = getattr(args, "slime_api_port", 18080)

        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=host,
            port=port,
            thread_name="dataagent-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        self.adapter_url = f"http://127.0.0.1:{self.app_handle.port}"
        logger.info(
            "DataAgentAdapter: listening on %s (sglang=%s tool_parser=%s)",
            self.adapter_url,
            sglang_url,
            tool_parser,
        )
```

### 3.2 重构 `custom_generate.py`

核心变化：
- `slime_api.ensure_server(args)` → `AdapterService(args)` (单例，自动启动)
- `_sample_map[thread_id] = sample` → `adapter.open_session(thread_id, ...)`
- 手动 token 管理 → 适配器自动管理
- `sample.reward = score(...)` → `adapter.finish_session(thread_id, reward=score(...))`
- 返回单个 Sample → 返回 `list[Sample]`（DataAgent 线性对话 → `[sample]`，与当前一致）

```python
"""Custom generate function bridging Slime to DataAgent via SSE.

Phase 3: fully leverages slime's OpenAIAdapter for LLM interception.
DataAgent calls the adapter's /v1/chat/completions instead of a hand-rolled proxy.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from examples.dataagent.adapter import AdapterService
from examples.dataagent.reward_func import score

logger = logging.getLogger(__name__)

# ── config ────────────────────────────────────────────────────────────

DATAAGENT_BASE_URL = os.environ.get("DATAAGENT_BASE_URL", "http://localhost:8065")
DATAAGENT_TIMEOUT = int(os.environ.get("DATAAGENT_TIMEOUT", "600"))


# ── SSE consumer ──────────────────────────────────────────────────────

async def _query_dataagent(query: str, thread_id: str) -> dict[str, Any]:
    """Send query to DataAgent, collect SSE events, return structured result.

    DataAgent internally makes LLM calls to the adapter's /v1/chat/completions.
    Each call carries X-DataAgent-Thread-Id → adapter routes to correct session.
    """
    url = f"{DATAAGENT_BASE_URL}/api/stream/search"
    params = {
        "agentId": int(os.environ.get("DATAAGENT_AGENT_ID", "20")),
        "query": query,
        "threadId": thread_id,
    }
    headers = {"Accept": "text/event-stream"}

    nodes: dict[str, dict[str, Any]] = {}
    node_order: list[str] = []
    error_msg: str | None = None

    async with httpx.AsyncClient(timeout=httpx.Timeout(DATAAGENT_TIMEOUT)) as client:
        async with client.stream("GET", url, params=params, headers=headers) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if not payload_str or payload_str == "[DONE]":
                    continue
                try:
                    d = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue

                if d.get("error"):
                    error_msg = d.get("text", "unknown")
                    break
                if d.get("complete"):
                    break

                node_name = d.get("nodeName", "unknown")
                text_type = d.get("textType", "TEXT")
                text_chunk = d.get("text", "")
                key = f"{node_name}|{text_type}"

                if key not in nodes:
                    nodes[key] = {"node": node_name, "type": text_type, "text": ""}
                    node_order.append(key)
                nodes[key]["text"] += text_chunk

    if error_msg:
        raise RuntimeError(f"DataAgent query failed: {error_msg}")

    return {"nodes": [nodes[k] for k in node_order]}


# ── slime entrypoint ───────────────────────────────────────────────────

async def generate(args: Any, sample: Any, sampling_params: dict) -> list:
    """Slime custom generate entrypoint.

    Called by generate_and_rm → generate_and_rm_group for each sample.
    Uses the standard adapter session lifecycle:

        open_session → [DataAgent calls adapter] → finish_session

    Returns list[Sample] for slime's generate_and_rm contract.
    DataAgent's linear conversation produces exactly 1 Sample
    (all turns concatenated with correct loss_mask by TrajectoryManager).
    If the conversation branches (sub-agents, compaction forks), the
    TrajectoryManager automatically fans out into multiple Samples.
    """
    from slime.utils.types import Sample

    # Start adapter (idempotent singleton).
    svc = AdapterService(args)
    adapter = svc.adapter

    # Session ID: deterministic, maps to X-DataAgent-Thread-Id header.
    thread_id = f"slime-{sample.index}"
    adapter.open_session(
        thread_id,
        sampling_defaults=sampling_params,
    )

    query = str(sample.prompt).strip()
    logger.info("DataAgent generate: threadId=%s query=%r", thread_id, query[:120])

    try:
        result = await _query_dataagent(query, thread_id)
    except Exception:
        logger.exception("DataAgent generate failed: threadId=%s", thread_id)
        await adapter.drop_session(thread_id)
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        return [sample]

    # Compute reward and close session → TrajectoryManager linearizes
    # the multi-turn tree into list[Sample] with correct loss masks.
    reward = score(result["nodes"], getattr(sample, "label", ""))
    samples = await adapter.finish_session(
        thread_id,
        base_sample=sample,
        reward=float(reward),
        extra_metadata={
            "dataagent_nodes": result["nodes"],
            "dataagent_thread_id": thread_id,
        },
    )

    if not samples:
        # Edge case: no turns recorded (e.g., DataAgent answered without LLM calls)
        sample.reward = reward
        sample.status = Sample.Status.COMPLETED
        sample.metadata = {
            **(sample.metadata or {}),
            "dataagent_nodes": result["nodes"],
            "dataagent_thread_id": thread_id,
        }
        return [sample]

    logger.info(
        "DataAgent generate: done (threadId=%s, nodes=%d, reward=%.2f, segments=%d)",
        thread_id,
        len(result["nodes"]),
        reward,
        len(samples),
    )
    return samples
```

### 3.3 `slime_api.py` —— 删除

整个文件的功能由 `OpenAIAdapter` + `AdapterService` 接管，可完全删除。

`slime_api.py` 中的 `notify_resume()` 逻辑转移到 adapter 内部的 abort 重试机制（见第 4 节）。

### 3.4 `reward_func.py` —— 保持不变

评分函数保持为纯函数。可选优化：通过 `--custom-rm-path examples.dataagent.reward_func.score` 注册，实现 generation 和 reward 的解耦。但如果 reward 需要访问 SSE 返回的 `nodes` 信息（而非仅 Sample 内的数据），则保持在 `custom_generate.py` 中调用更合理。

## 4. Abort/Resume 特殊处理

### 4.1 问题

DataAgent 是一个外部 Java 进程，通过 HTTP 调用 LLM。当 SGLang 进行权重同步（`pause_generation`）时：

- **coding_agent_rl**：适配器返回错误给沙箱内的 Claude Code，harness 层处理重试
- **dataagent 当前做法**：`slime_api.py` 检测 `finish_reason=abort`，保存部分 token，等待 `notify_resume()`，然后用 `input_ids + 部分 token` 重新调用 SGLang，**对 DataAgent 完全透明**

### 4.2 方案选择

| 方案 | 复杂度 | DataAgent 体验 | 推荐 |
|------|--------|---------------|------|
| A. 接受 abort 穿透 | 零代码 | DataAgent 看到截断响应，可能失败 | ❌ |
| B. 在 adapter 子类中覆盖 abort 处理 | ~30 行 | 完全透明 | ✅ |
| C. 修改 slime 框架提供 abort hook | 需改核心代码 | 可配置 | 长期 |

### 4.3 推荐实现：方案 B

在 `DataAgentAdapter` 中覆盖 `_run_turn` 方法，核心改动是将：

```python
turn = await call_sglang_generate(prompt_ids, ...)
```

替换为带 abort 重试的版本：

```python
async def _call_sglang_with_retry(self, prompt_ids, session, body, sid):
    """Call SGLang with transparent abort→wait→resume retry.

    When SGLang returns finish_reason=abort (weight sync), save partial
    tokens, wait for continue_generation + notify_resume, then re-invoke
    with the full context so DataAgent sees one coherent response.
    """
    while True:
        turn = await call_sglang_generate(
            prompt_ids, session, body, adapter=self, session_id=sid
        )

        if turn.finish_reason != "abort":
            return turn

        # Abort detected: save partial tokens for context continuity.
        if turn.output_ids:
            prompt_ids = list(prompt_ids) + list(turn.output_ids)

        # Wait for weight sync to complete.
        await self._wait_resume()
        # Loop: re-invoke SGLang with full (original + partial) context.
```

`_wait_resume()` 通过监听 `GenerateState.aborted` 标志的变化来实现，而不是 `slime_api.py` 中的计数器轮询：

```python
async def _wait_resume(self) -> None:
    """Block until SGLang resumes generation after weight sync."""
    from slime.rollout.sglang_rollout import GenerateState
    state = GenerateState(None)  # singleton, args not needed for reading .aborted
    while state.aborted:
        await asyncio.sleep(0.1)
```

注意：这里需要把 `call_sglang_generate` 变成可重试的。当前代码在 abort 时不会抛异常，而是正常返回 `TurnRecord`（带 `finish_reason=abort`）。所以重试逻辑只需检测 `finish_reason` 即可。

### 4.4 关于 `notify_resume`

当前 `slime_api.py` 中 `notify_resume()` 在 `update_weights()` 后被调用。使用 adapter 后，`_wait_resume()` 直接观察 `GenerateState.aborted`，不再需要外部的 `notify_resume()` 信号。`GenerateState.aborted` 在 `abort()` 函数中被设为 `True`，在 slime 的 `_continue_rollout` 流程中被重置。这个生命周期已经内置在 slime 的 `fully_async_rollout` 流程中。

## 5. Token 管理对比

### 5.1 当前做法（手动）

```python
# slime_api.py: handle_chat_completions()
if sample.response_length == 0:
    sample.tokens = (sample.tokens or []) + list(input_ids)
else:
    new_prompt_tokens = input_ids[len(sample.tokens):]
    if new_prompt_tokens:
        sample.append_response_tokens(_args, tokens=new_prompt_tokens, trainable=False)

# ... after SGLang response ...
new_tokens = _extract_response_tokens(output)
sample.append_response_tokens(_args, tokens=new_tokens, log_probs=..., trainable=True)
```

问题：
- 手动判断第一个 turn vs 后续 turn
- 手动管理 prompt/output token 的 loss_mask
- 不支持子代理分支、compaction 等高级模式

### 5.2 改造后（TrajectoryManager 自动管理）

```python
# adapter._run_turn() 内部自动处理（每次 DataAgent LLM 调用 = 1 个 turn）：
# 1. _render_token_ids() → prompt_ids
# 2. call_sglang_generate() → TurnRecord(prompt_ids, output_ids, log_probs)
# 3. TrajectoryManager.record_turn(sid, turn, prompt_messages, response_message)
#    - 将本轮 output token 追加到 _SampleBuilder，loss_mask=1（训练）
#    - 将本轮 prompt tail（新增的 system/user/tool 消息）追加，loss_mask=0
#    - 首轮 prompt 通过 leading_prompt_len 从训练区域剔除
#
# 4. finish_session() 时 _chain_to_samples() → 1 个 Sample（线性对话）
#    - _SampleBuilder 将所有 turn 拼接: [prompt_tail_1, output_1, prompt_tail_2, output_2, ...]
#    - loss_mask: [0,0,..., 1,1,..., 0,0,..., 1,1,...]
#    - 首轮 prompt 不进入训练区域（被 leading_prompt_len 跳过）
```

### 5.3 当前代码的一个 loss_mask bug

```python
# slime_api.py 第 113-114 行（首轮）：
if sample.response_length == 0:
    sample.tokens = (sample.tokens or []) + list(input_ids)
# ↑ 此时 loss_mask 为 None，append_response_tokens 会设 loss_mask = [1]*N
# 这意味着首轮 prompt token 全部 loss_mask=1 → 会被训练！
```

`TrajectoryManager` 通过 `_SampleBuilder.leading_prompt_len` 自动跳过首轮 prompt，不会产生这个 bug。

## 6. 改造收益

| 维度 | 改造前 | 改造后 |
|------|--------|--------|
| 代码量（核心） | `custom_generate.py` 139行 + `slime_api.py` 264行 = **403行** | `custom_generate.py` ~100行 + `adapter.py` ~60行 = **~160行** |
| LLM 代理实现 | 手动 aiohttp + SGLang 调用 | 继承 `OpenAIAdapter`（slime 框架维护） |
| 多轮对话管理 | 手动管理首轮/后续 turn 的 token 追加 | `TrajectoryManager` + `_SampleBuilder` 自动拼接，首轮 prompt 自动剔除 |
| loss_mask 正确性 | 手动设置，首轮 prompt token 存在 loss_mask=1 的 bug | 框架自动：prompt=0, output=1, 首轮 prompt 通过 leading_prompt_len 跳过 |
| 子代理/分支 | 不支持 | `TrajectoryManager` 自动 fork/merge（DataAgent 线性对话不会触发） |
| Abort/Resume | 手动计数器轮询 | 观察 `GenerateState` 状态（更可靠） |
| aiohttp 启动 | 手动 thread + event loop | `run_app_in_thread`（含 handler_cancellation） |
| 工具调用解析 | 不支持 | `parse_model_output` + tool_parser 自动解析 |
| 子代理/分支 | 不支持 | `TrajectoryManager` 自动 fork/merge |
| 框架升级受益 | ❌ 手动维护 | ✅ 框架改进自动获得（bug 修复、新功能） |

## 7. 实施步骤

### Step 1: 新增 `adapter.py`

创建 `DataAgentAdapter(OpenAIAdapter)` 子类 + `AdapterService` 单例，约 60 行。核心改动点：
- 重写 `_session_id` 读取 `X-DataAgent-Thread-Id`
- 添加 `_call_sglang_with_retry` 处理 abort 重试

### Step 2: 重构 `custom_generate.py`

- 删除所有 `slime_api` 引用
- 引入 `AdapterService` + `adapter.open_session/finish_session`
- 返回类型改为 `list[Sample]`

### Step 3: 更新启动脚本

`run_qwen3_14B_fully_async.sh` 中：
- 删除 `notify_resume` 相关逻辑（如果有）
- 确保 `sglang_tool_call_parser` 参数正确传递（如果 DataAgent 使用工具调用）
- `PYTHONPATH` 确保包含 `examples/dataagent`

### Step 4: 更新测试

- `test_abort_resume.py`：改用 adapter session 生命周期
- `check_output.py`：改用新的 `generate()` 签名（返回 `list[Sample]`）

### Step 5: 删除 `slime_api.py`

确认所有功能正常后删除。

### Step 6: (可选) 将 reward 注册为标准 RM

```bash
--custom-rm-path examples.dataagent.reward_func.score
```

如果 reward 不需要访问 DataAgent SSE 返回的 `nodes` 信息（仅基于 Sample 的 response/metadata），则可以完全解耦。

## 8. 风险与注意事项

1. **DataAgent 的 LLM 请求格式**：需确认 DataAgent (Spring AI) 发送的请求是否完全符合 OpenAI Chat Completions 规范。`OpenAIAdapter` 支持标准字段（`messages`, `tools`, `stream`, `temperature` 等），如果 DataAgent 使用了非标准字段，需要在子类中处理。

2. **DataAgent 的线性对话不会 fan-out**：DataAgent 内部 LLM 调用是线性累积的（无子代理、无分支），`TrajectoryManager` 产出 1 个 Sample。与当前行为一致。如果未来 DataAgent 支持并行子任务导致分支，`finish_session` 会自动产出多个 Sample，reward 平均分配。

3. **Abort 期间的部分 token**：重试时使用 `input_ids + 部分 token` 作为新的 input_ids。这会导致部分 token 的 loss_mask=0（非训练），只有最终完成的 token 才被训练。回退兼容通过 `--partial-rollout` + `--mask-offpolicy-in-partial-rollout` 处理。

4. **`generate_and_rm` 的 reward 重复赋值**：在 `custom_generate` 返回时，`sample.reward` 已经设置。`generate_and_rm` 会检查 `sample.reward is None`，如果已设置则跳过 RM 调用。这与当前行为一致。

5. **并发安全性**：`AdapterService` 是进程级单例（`SingletonMeta`），多个并发的 `generate()` 调用共享同一个 adapter 实例。`BaseAdapter` 的 session 管理是线程安全的（每个 session 独立的 trajectory tree + inflight task set）。

## 9. 与 coding_agent_rl 的架构对齐

改造后，dataagent 和 coding_agent_rl 的架构完全对齐：

```
┌─────────────────────────────────────────────────────────┐
│                    coding_agent_rl                       │
│                                                         │
│  generate()                                             │
│    ├── _AdapterService (AnthropicAdapter)               │
│    ├── boot_agent_sandbox()                             │
│    ├── swe.prepare_workspace()                          │
│    ├── HARNESS.run()  →  adapter 拦截 LLM 调用           │
│    ├── swe.git_diff()                                   │
│    ├── swe.evaluate()                                   │
│    └── adapter.finish_session(reward=) → list[Sample]   │
│              (分支场景: N 个 Sample; 线性对话: 1 个)       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    dataagent (改造后)                     │
│                                                         │
│  generate()                                             │
│    ├── AdapterService (DataAgentAdapter)                │
│    ├── _query_dataagent()  →  adapter 拦截 LLM 调用      │
│    ├── score(nodes)                                     │
│    └── adapter.finish_session(reward=) → list[Sample]   │
│              (DataAgent 线性对话 → 1 个 Sample)           │
└─────────────────────────────────────────────────────────┘
```

两者都是：**启动适配器 → 运行 Agent（其 LLM 调用被适配器自动拦截）→ 打分 → finish_session**。

> **关于「多段 fan-out」的纠正**：DataAgent 的 8~12 轮 LLM 调用是**线性累积**的（同一棵对话树，无分支），`TrajectoryManager.get_trajectory()` 产出的仍是 **1 个 Sample**。多个 Sample 只在出现分支时才产生（如 coding_agent_rl 的子代理 fork、compaction 导致的 re-tokenization drift → FORK）。改造的真正收益是 **loss_mask 正确性**（见第 5 节），而非 fan-out。
