# Agent Fully Async 设计文档

## 1. 问题定义

SearchAgent-Zero 已有三层 async，Agent 内部仍是串行阻塞：

| 层级 | 位置 | 描述 | 状态 |
|------|------|------|------|
| Agent Loop 内部 | `tool_agent_loop.py` | 单样本内 asyncio，同 turn 内 tool calls 并行 | 已有 |
| Worker 级别 | `agent_loop.py` | 多样本间 `asyncio.create_task` + `asyncio.gather` | 已有 |
| 训练级（单步 Fully Async） | `fully_async_policy/` | Rollouter/Trainer 解耦为独立 Ray Actor + MessageQueue | 已有 |
| **Agent 级 Fully Async** | — | LLM 生成与 Tool 执行解耦（流式检测 + 提前调度） | **待开发** |

当前 `ToolAgentLoop` 单步是串行的：

```
GENERATING（等 LLM 完整输出）→ PROCESSING_TOOLS（等所有 tool 执行完）→ GENERATING → ...
```

每步延迟 = LLM 生成时间 + tool 执行时间，两者无法重叠。Agent 级 Fully Async 的目标是把它们变成流水线：

```
时间线（当前）：|-- LLM 生成 --|-- Tool1+Tool2 --|-- LLM 生成 --|
时间线（目标）：|-- LLM 流式 --|  边生成边检测到 tool call，立即调度
                |-- Tool1(提前) --|
                  |-- Tool2 --|  ← 不等同轮其他 tool
                    |-- LLM(结果注入就继续) --|
```

---

## 2. 增量开发路径

### 2.1 架构优势：Agent Loop 天然可插拔

项目的 Agent Loop 使用 `@register` + Hydra 动态实例化模式，切换 agent 类型只需改一行配置：

```python
# AgentLoopWorker._run_agent_loop() — agent_loop.py:615-647
agent_loop_config = _agent_loop_registry[agent_name]     # 按名查找
agent_loop = hydra.utils.instantiate(                    # 动态实例化
    config=agent_loop_config,
    server_manager=self.server_manager,                  # 注入 server
    tokenizer=self.tokenizer,
    processor=self.processor,
)
output = await agent_loop.run(sampling_params, **kwargs)  # 统一接口
```

运行时切换：

```bash
# 当前
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent

# 新 StreaminAgentLoop（开发完成后）
actor_rollout_ref.rollout.agent.default_agent_loop=streaming_tool_agent
```

`server_manager` 同样可替换（`FullyAsyncLLMServerManager` 就是通过覆盖 `AgentLoopWorker.__init__` 注入的）。这构成了完全向后兼容的扩展基础。

### 2.2 五步增量计划

#### Step 1：Streaming LLM 接口（不影响现有逻辑）

**改动文件**：`verl/experimental/agent_loop/agent_loop.py`

```python
class AsyncLLMServerManager:
    # 现有接口不变
    async def generate(self, request_id, prompt_ids, ...) -> TokenOutput: ...

    # 新增
    async def generate_stream(
        self, request_id, prompt_ids, ...
    ) -> AsyncIterator[TokenChunk]:
        """流式 yield token chunk，保持 sticky session 和 load balancing。"""
        server_id, server = await self._acquire_server(request_id)
        try:
            async for chunk in server.generate_stream.remote(...):
                yield chunk
        finally:
            self._release_server(server_id)
```

`FullyAsyncLLMServerManager` 同步适配：外层 `while True` 分段恢复循环不变，内层从 `generate()` 换成 `generate_stream()`，逐 chunk yield。

#### Step 2：增量 Tool Parser（不影响现有 `extract_tool_calls`）

**改动文件**：`verl/experimental/agent_loop/tool_parser.py`

```python
class ParseResult:
    INCOMPLETE = "incomplete"        # 继续接收 token
    TOOL_CALL_READY = "tool_call_ready"  # 检测到完整 tool call
    TEXT_DONE = "text_done"          # 纯文本生成完成

class ToolParser(ABC):
    # 现有接口不变
    async def extract_tool_calls(self, response_ids: list[int], tools) -> tuple[str, list[FunctionCall]]: ...

    # 新增增量接口
    def try_parse_partial(self, accumulated_text: str) -> ParseResult:
        """对已累积的文本尝试检测 tool call。不阻塞，同步返回。"""
```

3 个 parser（Hermes / GptOss / Qwen3XML）各自实现 `try_parse_partial`。以 Hermes（`ASearch` 实际使用的格式）为例：

```
流式文本累积：  "some reasoning...<tool_call>{"query_list": ["what is
检测点：        ✗ INCOMPLETE（JSON 没闭合）

继续累积：      "some reasoning...<tool_call>{"query_list": ["what is AI"]}</tool_call>"
检测点：        ✓ TOOL_CALL_READY → FunctionCall(name="search", arguments={...})
```

Hermes 的 XML 标签格式对增量解析最友好 — 遇到 `</tool_call>` 闭合标签就能确定一个完整 tool call，不像纯 JSON 需要处理括号匹配。

#### Step 3：注册 `StreamingToolAgentLoop`

**新建文件**：`verl/experimental/agent_loop/streaming_tool_agent_loop.py`（~200 行）

```python
@register("streaming_tool_agent")
class StreamingToolAgentLoop(ToolAgentLoop):
    """继承 ToolAgentLoop，只重写 run() 和 _handle_generating_state()。"""

    async def run(self, sampling_params, **kwargs) -> AgentLoopOutput:
        # 同父类，但状态机调度中加入 STREAMING_GENERATING 分支
        ...

    async def _handle_streaming_generating_state(
        self, agent_data, sampling_params
    ) -> AgentState:
        """核心变更：用 generate_stream + try_parse_partial 替代 generate + extract_tool_calls。"""
        dispatch_tasks = []
        accumulated_text = ""

        async for chunk in self.server_manager.generate_stream(...):
            agent_data.prompt_ids.extend(chunk.token_ids)
            accumulated_text += self.tokenizer.decode(chunk.token_ids)

            result = self.tool_parser.try_parse_partial(accumulated_text)
            if result == ParseResult.TOOL_CALL_READY:
                # 立即调度，不等 stream 结束
                for tc in self._extract_tool_calls_from_text(accumulated_text):
                    task = asyncio.create_task(self._call_tool(tc, ...))
                    dispatch_tasks.append(task)

            if chunk.finish_reason:
                break

        if dispatch_tasks:
            results = await asyncio.gather(*dispatch_tasks)
            self._inject_tool_results(agent_data, results)
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    # 以下方法直接从父类继承，零改动：
    # _handle_pending_state()          ✅
    # _handle_processing_tools_state() ✅
    # _call_tool()                     ✅
    # _truncate_text_by_tokens()      ✅
    # summarization 逻辑               ✅
    # 异常检测                         ✅
```

#### Step 4：注册 YAML 配置

**新建文件**：`examples/search_agent_rl/config/agent_loop/streaming_tool_agent.yaml`

```yaml
- name: streaming_tool_agent
  _target_: verl.experimental.agent_loop.streaming_tool_agent_loop.StreamingToolAgentLoop
```

#### Step 5：与 Fully Async 训练组合

```bash
# 在 ASearch_fully_async.sh 基础上，只改一行：
actor_rollout_ref.rollout.agent.default_agent_loop=streaming_tool_agent
```

`FullyAsyncLLMServerManager` 在 Step 1 已适配 streaming，`while True` 外层循环不变。

### 2.3 改动量总结

| 类别 | 文件 | 改动量 |
|------|------|--------|
| 新建 | `streaming_tool_agent_loop.py` | ~200 行 |
| 新建 | `streaming_tool_agent.yaml` | ~3 行 |
| 修改 | `agent_loop.py`（`AsyncLLMServerManager`） | +~60 行 |
| 修改 | `tool_parser.py`（3 个 parser） | +~200 行 |
| 修改 | `fully_async_policy/agent_loop/agent_loop.py` | +~40 行 |
| **合计** | | **~500 行新增，0 行删除** |

对比训练级 fully async（2,774 行全新代码），Agent async 开发量约为 **1/5**，因为它复用了父类 90% 的逻辑（pending 处理、tool 执行、summarization、异常检测、response 拼接全部继承）。

---

## 3. 通用性分析

Agent 级 Fully Async 是通用模式，不绑定 SearchAgent。把它放到更抽象的层次看：

### 3.1 所有 Tool-using Agent 的共性

任何 tool-using agent 的时间线都是：

```
Turn N: LLM 生成 → Tool 执行 → 结果注入 prompt
Turn N+1: LLM 生成 → Tool 执行 → 结果注入 prompt
...
```

两个阻塞点在所有 agent 类型里都存在：

| 阻塞点 | 等什么 | 影响范围 |
|--------|--------|---------|
| LLM 生成 → Tool 执行 | 必须知道调什么 tool 才能执行 | 所有 agent |
| Tool 执行 → LLM 生成 | 必须拿到结果才能构建 prompt | 所有 agent |

Agent fully async 的本质就是打破这两个"必须等"。

### 3.2 不同 Agent 类型的差异

| Agent 类型 | Tool | 生成→执行优化空间 | 执行→生成优化空间 |
|-----------|------|-------------------|-------------------|
| SearchAgent | 检索 API（~100ms-2s） | 流式检测 search query | 结果分段注入 |
| Code Agent | 沙盒执行（~1s-30s） | 检测到 bash 命令立刻执行 | 执行输出流式回传 |
| Computer Use | GUI 操作（~100ms-500ms） | 检测到 action 立刻执行 | 截图异步返回 |
| Math Agent | 无 tool | 无优化空间 | 无优化空间 |

差异只在于 tool 执行耗时和结果大小，流式调度模式完全一样。

### 3.3 通用分层设计

```
┌──────────────────────────────────────────────────────────┐
│  通用层（agent_loop/ 下，所有 agent 复用）                 │
│                                                          │
│  StreamingToolAgentLoopBase (ABC)                        │
│  ├── generate_stream()        流式 LLM 输出              │
│  ├── IncrementalToolParser    增量解析抽象接口            │
│  ├── _streaming_dispatch()    并发调度骨架                │
│  ├── AgentData.snapshot()     并发安全快照                │
│  └── _critical_section()      可中断性控制                │
│                                                          │
├──────────────────────────────────────────────────────────┤
│  Agent 特化层（只注入 parser 和 executor）                 │
│                                                          │
│  @register("streaming_search_agent")                     │
│  class StreamingSearchAgent(StreamingToolAgentLoopBase):  │
│      tool_parser = HermesToolParser()                    │
│      # 其余全部继承                                       │
│                                                          │
│  @register("streaming_code_agent")                       │
│  class StreamingCodeAgent(StreamingToolAgentLoopBase):    │
│      tool_parser = BashToolParser()                      │
│      # 其余全部继承                                       │
└──────────────────────────────────────────────────────────┘
```

建议 Phase 1 做完 SearchAgent 特化后，把通用部分抽到 `StreamingToolAgentLoopBase`，让后续 agent 类型可直接复用。

---

## 4. Partial Rollout 兼容设计

Partial rollout 是训练级 async 的核心机制：Trainer 同步新权重 → LLM 生成被中断 → 用新权重从断点恢复。Agent streaming 使其更复杂，因为 tool 可能已在"执行中"。

### 4.1 三种中断场景

```
权重同步触发点

场景 A：纯生成阶段（无 tool call 检测）
│── LLM streaming ──│ 中断 │── v2 恢复生成 ──│
✅ 没问题。只有 token，无副作用。FullyAsyncLLMServerManager 的 while True 完全能处理。

场景 B：tool call 已检测、已 dispatch、执行中
│── LLM streaming ──│ tc! │── Tool 执行中... ──│ 中断 │
                     │                            │
                     └─ asyncio task running ──────┘
❌ 需要策略：等待完成 vs 取消丢弃

场景 C：tool 结果已注入，正在下一轮生成
│── v1 tc → Tool完成 → 结果注入prompt → LLM v1 生成中 ──│ 中断 │
❌ prompt 混入了 v1 权重的 tool 结果，v2 继续的语义不一致
```

### 4.2 推荐策略：Commit at Detection

项目已通过 `staleness_threshold=0.5` 选择了"接受一定不一致"的路线。Agent streaming 沿袭这一选择：

| 中断点 | 策略 |
|--------|------|
| 纯生成阶段 | ✅ 允许中断，token 累积后恢复 |
| tool call 已 dispatch | ⚠️ 等待完成（~1-2s），不取消，不浪费 |
| tool 结果已注入 | ⚠️ 不回滚，v2 在此基础上继续，staleness_tracker 正常标记 |

### 4.3 实现：分阶段推进

**Phase 1（推荐首选）**：Turn 级安全点

只在 turn 边界允许 partial rollout，整个 streaming turn 内不中断。

```python
# FullyAsyncLLMServerManager.generate_stream()
async def generate_stream(self, request_id, prompt_ids, ...):
    final_output = TokenOutput(token_ids=[], ...)
    
    while True:  # 外层：partial rollout 恢复循环
        in_turn = True
        
        async for chunk in super().generate_stream(
            prompt_ids=prompt_ids + final_output.token_ids, ...
        ):
            final_output.token_ids.extend(chunk.token_ids)
            yield chunk
            
            if self._tool_call_detected(chunk):
                in_turn = True  # 进入临界区
        
        if not in_turn:
            if self._should_preempt(request_id):
                continue  # 安全点：回到 while True，用新权重恢复
            else:
                break
        else:
            break  # tool call 已检测，让 agent loop 处理完
    
    return final_output
```

**Phase 2**：细粒度中断

在"纯生成，且至少产生 N 个 token 后"允许中断，tool 执行中不允许。需要 `in_critical_section` 信号。

### 4.4 与 Agent Loop 的交互

```python
# Agent 层保持简单，不感知 partial rollout
async def _handle_streaming_generating_state(self, agent_data, sampling_params):
    dispatch_tasks = []
    snapshot = agent_data.snapshot()  # 保险用
    
    async for chunk in self.server_manager.generate_stream(...):
        agent_data.prompt_ids.extend(chunk.token_ids)
        
        result = self.tool_parser.try_parse_partial(accumulated_text)
        if result == ParseResult.TOOL_CALL_READY:
            for tc in result.tool_calls:
                task = asyncio.create_task(self._call_tool(tc, ...))
                dispatch_tasks.append(task)
    
    # tool call 已 dispatch → 已提交，即使此时权重同步也不回滚
    if dispatch_tasks:
        results = await asyncio.gather(*dispatch_tasks)
        self._inject_tool_results(agent_data, results)
    
    return AgentState.PROCESSING_TOOLS
```

Partial rollout 的复杂性完全封装在 `FullyAsyncLLMServerManager.generate_stream()` 内。Agent loop 看到的就是一个正常的流，偶尔 token chunk 之间有"暂停"（权重同步），对它透明。

---

## 5. Multi-Agent / Sub-Agent 架构扩展

加上 sub-agent 和 multi-agent 后，async 的抽象对象从 tool call 升级为 agent 本身。

### 5.1 当前架构的根本局限

当前 Agent 是"一次性协程"：

```python
async def run(self, ...) -> AgentLoopOutput:
    state = PENDING
    while state != TERMINATED:
        # 三个状态分支
    return output  # 唯一出口
```

问题：
- `run()` 从开始到结束拥有完整控制流
- 没有"等待外部消息"的状态
- 无法在中间暂停、接收 sub-agent 结果、再恢复

### 5.2 并发模型升级

```
Phase 1（单 Agent）：并发单位 = tool call

  Agent ──→ Tool1 ──→ Tool2 ──→ Tool3
           └── 并行 ──┘

Phase 3（Sub-agent）：并发单位 = Agent（每个 agent 有自己的 tool 树）

                  ┌── SubA ──→ Tool_a1 ──→ Tool_a2 ──→ 结果 ──┐
  Main ──→ tc ───┤                                            │──→ 汇总 → 继续
                  └── SubB ──→ Tool_b1 ──→ 结果 ──────────────┘
                  
  Agent 间并行 + 每个 Agent 内部也并行 → 两层嵌套 async
```

### 5.3 通用架构设计

核心思想：**Agent 从"一次性协程"升级为"可挂起、可通信的实体"**。

```
┌──────────────────────────────────────────────────────────┐
│  Agent Runtime（新增，管理 Agent 生命周期和通信）           │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ Agent A  │  │ Agent B  │  │ Agent C  │  Agent 实例    │
│  │ (main)   │  │ (sub)    │  │ (sub)    │  各自独立状态机 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘               │
│       │              │              │                     │
│       └──────┬───────┴──────┬───────┘                     │
│              │              │                             │
│  ┌───────────┴──────────────┴────────────┐               │
│  │         Resource Scheduler            │  共享资源调度  │
│  │   LLM servers / Tool instances        │               │
│  └──────────────────────────────────────┘               │
│                                                          │
│  ┌──────────────────────────────────────┐               │
│  │         Message Bus                  │  Agent 间通信  │
│  │   spawn / result / cancel / broadcast│               │
│  └──────────────────────────────────────┘               │
└──────────────────────────────────────────────────────────┘
```

### 5.4 核心抽象变更

```python
class AgentState(Enum):
    IDLE = "idle"                   # 等待分配任务
    GENERATING = "generating"       # LLM streaming 生成
    EXECUTING_TOOLS = "executing_tools"
    WAITING_CHILDREN = "waiting_children"  # ← 新增：等 sub-agent 结果
    TERMINATED = "terminated"

class Agent:
    """通用 Agent — 单 agent 是多 agent 的特例。"""
    
    agent_id: str
    parent_id: str | None
    children: dict[str, "Agent"]
    
    # 可被外部驱动的状态机入口（替代 run()）
    async def step(self, message: AgentMessage | None) -> AgentMessage | None:
        """执行一步状态机转换。
        
        单 agent 场景：外界不断调 step(None) 直到 TERMINATED
        多 agent 场景：外界在 agent 之间轮转 step()
        """
    
    async def run(self, ...) -> AgentLoopOutput:
        """退化为 step() 的 while 循环包装器（向后兼容）。"""
        while self.state != AgentState.TERMINATED:
            await self.step(None)
        return self._build_output()

# 状态机对比
# 单 Agent（现在）：PENDING → GENERATING ↔ PROCESSING_TOOLS → TERMINATED
# 单 Agent（async）：同上，GENERATING 内部 streaming + 提前调度
# Multi-Agent（新）：            ┌→ EXECUTING_TOOLS ──┐
#   IDLE → GENERATING ──→ WAITING_CHILDREN ──→ GENERATING → TERMINATED
#                        └→ (self-terminate) ────────────────┘
```

### 5.5 Streaming 在 Multi-agent 下的角色

```
Main Agent 的 streaming 生成流中检测到 spawn sub-agent：

│── Main LLM streaming ──│ spawn_A! │ spawn_B! │── 继续生成 ──│
                             │           │
                             ▼           ▼
                       SubAgent A   SubAgent B
                       (独立stream) (独立stream)
                             │           │
                             ▼           ▼
                       结果异步回调 Main
                             │           │
                             └─────┬─────┘
                                   ▼
                          Main 继续 → 汇总

Main 不阻塞等 sub-agent — spawn 后继续自己的生成
Sub-agent 结果通过 MessageBus 异步注入
```

### 5.6 三层 Async 组合全景

```
层级 3（训练）：  Rollouter ←── async ──→ Trainer
                              │ 权重同步 = 中断源
                              │
层级 2（Agent 间）：Main ←── async ──→ SubA, SubB
                              │ sub-agent 结果 = 回调
                              │
层级 1（Agent 内）：LLM streaming ←── async ──→ Tool 执行
                              │ token 流 = 提前调度
```

Partial rollout 在此全景下的决策树：

```
权重同步触发
  │
  ├── 当前有 sub-agent 在执行？
  │     ├── 是 → 等待完成 或 取消（取决于 staleness_threshold）
  │     └── 否 → 继续判断
  │
  ├── Main agent 处于 WAITING_CHILDREN？
  │     └── 是 → 安全点！在消息边界同步
  │
  ├── Main agent 处于 GENERATING（纯生成，无 tool call 提交）？
  │     └── 是 → 在 token chunk 边界中断
  │
  └── Main agent tool call 已提交？
        └── 是 → 进入临界区，推迟中断
```

---

## 6. 分阶段落地建议

| Phase | 内容 | 依赖 | 产出 |
|-------|------|------|------|
| **1** | 单 Agent streaming | 无 | `StreamingToolAgentLoop`，SearchAgent 可用 |
| **2** | Agent 可挂起 | Phase 1 | `Agent.step()` + WAITING 状态，`run()` 退化为 wrapper |
| **3** | Sub-agent | Phase 2 | spawn/await_children 原语，MessageBus，ResourceScheduler |
| **4** | Multi-agent patterns | Phase 3 | 预置模式（vote/debate/pipeline），Agent Group 抽象 |

每个 Phase 产出可独立测试，不阻塞下一阶段。Phase 1 做完即可在 SearchAgent 场景获得延迟收益；Phase 2 打开扩展点但不改变 Phase 1 的用户体验；Phase 3-4 按需推进。

---

## 7. 核心改动文件总览

| 文件 | Phase 1 | Phase 2 | Phase 3-4 |
|------|---------|---------|-----------|
| `agent_loop/agent_loop.py` | +`generate_stream()` | | |
| `agent_loop/tool_parser.py` | +`try_parse_partial()` | | |
| `agent_loop/streaming_tool_agent_loop.py` | **新建** | | |
| `agent_loop/agent.py` | | **新建** `Agent` 抽象类 | +`step()`, `spawn()` |
| `agent_loop/agent_runtime.py` | | **新建** | +调度器, MessageBus |
| `agent_loop/agent_group.py` | | | **新建** 预置模式 |
| `fully_async_policy/agent_loop/agent_loop.py` | `generate_stream()` 适配 | | |
| Config YAML | +`streaming_tool_agent.yaml` | | |
