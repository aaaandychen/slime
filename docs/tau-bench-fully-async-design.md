# Tau-Bench Fully Async 改造设计文档

## 1. 目标

将 `examples/tau-bench/` 从同步训练（`train.py` + `--colocate`）改造为 fully async 训练（`train_async.py` + 异步 rollout worker + staleness 控制），模型从 Qwen3-4B 升级到 Qwen3-14B，运行在单机 8xH200 上。

## 2. 现状分析

### 2.1 当前架构

```
train.py (同步)
  └─ generate_rollout (sglang_rollout.py)
       └─ generate_and_rm_group → generate_and_rm → generate_with_tau.generate()
            └─ TrainableAgentMixin.asolve()
                 └─ 多轮循环: _call_llm(post SGLang) → _parse_tool → _execute_tool(env.step)
```

关键特征：
- 每轮 `_call_llm` 直接 `post(url, {"text": ..., "sampling_params": ...})`，走 SGLang `/generate` 的 text 模式
- **不请求 `return_logprob`**，没有 rollout logprobs
- 通过 `_get_token_delta` 计算 token delta 和 loss_mask
- tau-bench 环境在 agent 内部闭环（初始化 → 多轮交互 → 返回 reward）
- reward 由 tau-bench 环境直接给出，不走 slime 的 `--custom-rm-path`

### 2.2 目标架构

```
train_async.py
  └─ generate_rollout_fully_async (fully_async_rollout.py)
       └─ AsyncRolloutWorker._loop() [后台线程 + asyncio]
            └─ generate_and_rm_group → generate_and_rm → generate_with_tau.generate()
                 └─ TrainableAgentMixin.asolve()  [同上，但需修复 logprobs]
       └─ _generate_rollout_async() [按需取 target 个，不全挖空]
  └─ update_weights() → RolloutManager.reset_staleness() → worker.reset_staleness()
```

## 3. GPU 分配方案

8xH200，每卡 141GB HBM。

```
ACTOR_GPUS=4    (训练: TP=2, PP=1, CP=1, EP=1)
ROLLOUT_GPUS=4  (推理: 2 个 engine 各 2 卡 TP=2)
```

Qwen3-14B 是 40 层 dense 模型，hidden=5120，~14B 参数。TP=2 下每卡 ~7B 参数 ≈ 14GB (FP16)，加上 optimizer states + activations，H200 141GB 非常宽裕。

## 4. 文件改动清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `examples/tau-bench/run_qwen3_14B_fully_async.sh` | **新增** | 启动脚本 |
| `examples/tau-bench/generate_with_tau.py` | 修改 | 加 logprobs、去 partial_rollout assert |
| `examples/tau-bench/trainable_agents.py` | 修改 | `_call_llm` 加 `return_logprob`、payload 改用 `input_ids` |
| `slime/rollout/fully_async_rollout.py` | 已完成 | staleness 控制 |
| `slime/ray/rollout.py` | 已完成 | reset_staleness 联动 worker |

---

## 5. 详细改动

### 5.1 启动脚本 `run_qwen3_14B_fully_async.sh`（新增）

基于 `examples/fully_async/run-qwen3-4B-fully_async.sh` + `examples/tau-bench/run_qwen3_4B.sh` 合并。

```bash
#!/bin/bash
# Tau-bench fully async training with Qwen3-14B on 8xH200.
set -ex

export PYTHONUNBUFFERED=1

# --- cleanup ---
pkill -9 sglang 2>/dev/null || true
sleep 3
ray stop --force 2>/dev/null || true
pkill -9 ray python 2>/dev/null || true
sleep 3

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-14B.sh"

# --- paths ---
MODEL_DIR=${MODEL_DIR:-/root/models/Qwen3-14B}
DATA_DIR=${DATA_DIR:-/root/tau-bench}

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_DIR}"
    --ref-load "${MODEL_DIR}_torch_dist"
    --save "${MODEL_DIR}_slime_async"
    --save-interval 20
)

ROLLOUT_ARGS=(
    # ↓↓↓ Fully async core ↓↓↓
    --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
    --update-weights-interval 2
    --staleness-threshold 1

    # ↓↓↓ Tau-bench specific ↓↓↓
    --custom-generate-function-path generate_with_tau.generate
    --prompt-data "${DATA_DIR}/retail_train_tasks.jsonl"
    --input-key index
    --rollout-shuffle

    --num-rollout 500
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    --rollout-max-response-len 1024
    --rollout-temperature 1.0

    --global-batch-size 128
    --balance-data
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

EVAL_ARGS=(
    --eval-interval 5
    --eval-prompt-data retail-dev "${DATA_DIR}/retail_dev_tasks.jsonl"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 1024
    --eval-top-k 1
)

PERF_ARGS=(
    --tensor-model-parallel-size 2
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1

    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1

    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static 0.7
    --sglang-server-concurrency 128
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --transformer-impl transformer_engine
    --log-passrate
)

# --- GPU split ---
NUM_GPUS=8
ACTOR_GPUS=4
ROLLOUT_GPUS=4

# --- launch ---
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}/../..\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"GEMINI_API_KEY\": \"${GEMINI_API_KEY:-}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_GPUS}" \
    --rollout-num-gpus "${ROLLOUT_GPUS}" \
    ${MODEL_ARGS[@]} \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}"
```

### 5.2 `trainable_agents.py` — _call_llm 加 logprobs

**问题**：`_call_llm` 的 payload 只传 `{"text": ..., "sampling_params": ...}`，不请求 `return_logprob`。GRPO 训练需要 `rollout_log_probs`。

**改动**：

```python
# _call_llm 方法改造
async def _call_llm(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    # 加 return_logprob
    payload["return_logprob"] = True
    return await post(url, payload)
```

同时 `asolve()` 中需要从 `output["meta_info"]["output_token_logprobs"]` 提取 logprobs，替代当前的 `_get_token_delta` 文本 delta 方案：

```python
# asolve() 中，原来：
#   response = output["text"]
#   assistant_token_ids, assistant_loss_mask = self._get_token_delta(state.tokenizer, messages)
#   response_token_ids.extend(assistant_token_ids)
#   loss_masks.extend(assistant_loss_mask)

# 改为：
if "output_token_logprobs" in output["meta_info"]:
    cur_response_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
    cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
else:
    # fallback: 没有 logprobs 时用 text tokenize（工具/环境 token 仍用此路径）
    cur_response_token_ids = state.tokenizer.encode(output["text"], add_special_tokens=False)
    cur_log_probs = None

# 工具/环境 token 仍用 _get_token_delta（不需要 logprobs，loss_mask=0）
```

**影响**：`_build_final_result` 需要新增 `rollout_log_probs` 字段，`InteractionResult` 也需要加。

### 5.3 `generate_with_tau.py` — generate 函数适配

**a) 去 `partial_rollout` assert**

第 217 行：
```python
# 改前
assert not args.partial_rollout, "Partial rollout is not supported..."
# 改后：允许 partial_rollout，abort 时标记 ABORTED 并让 fully async worker requeue
```

fully async 权重同步时 `pause_generation` 会 abort 进行中的请求。generate 函数已在 234-238 行处理 abort：
```python
if output["meta_info"]["finish_reason"]["type"] == "abort":
    res.status = Status.ABORTED
    return self._build_final_result(...)
```
返回的 `InteractionResult.status == ABORTED` → `Sample.status = ABORTED` → worker 的 `_make_done_cb` 检测 → requeue。

**b) requeue 后的重试**

generate 函数已有清理逻辑（222-228 行）：
```python
sample.rollout_log_probs = None
sample.response = ""
sample.response_length = 0
sample.loss_mask = []
```
requeue 后从头开始生成，tau-bench 环境重新初始化。

**c) reward 设置保持不变**

reward 由 tau-bench 环境在 agent 内部给出，`generate_and_rm` 检测到 `sample.reward is not None` 会跳过 `async_rm()`，不需要额外适配。

### 5.4 `generate_with_tau.py` / `InteractionResult` — 传递 logprobs

`InteractionResult` 新增字段：
```python
@dataclass
class InteractionResult:
    ...
    rollout_log_probs: list[float] | None = None  # 新增
```

`_build_final_result` 中设置：
```python
res.rollout_log_probs = accumulated_log_probs
```

`generate()` 函数中从 `InteractionResult` 搬运到 `Sample`：
```python
sample.rollout_log_probs = res.rollout_log_probs
```

### 5.5 已完成的改动（无需再改）

**`slime/rollout/fully_async_rollout.py`**：
- `AsyncRolloutWorker.__init__`：staleness 三字段
- `_loop()`：配额耗尽暂停拉取
- `_make_done_cb`：成功样本计数 +1
- `reset_staleness()`：计数器归零
- `get_completed_groups(max_groups=N)`：按需取不挖空
- `_generate_rollout_async`：`needed = target - len(collected)` 按需取

**`slime/ray/rollout.py`**：
- `reset_staleness()`：委托 `_global_worker.reset_staleness()`

---

## 6. ABORTED 样本完整链路

```
[生成中] asolve() 多轮循环中, _call_llm 等待 post()
    ↓
[权重同步] update_weights() → pause_generation.remote()
    ↓
post() 返回 finish_reason="abort"
    ↓
asolve() 检测 abort → res.status = ABORTED → _build_final_result()
    ↓
generate_with_tau.generate() → sample.status = ABORTED
    ↓
generate_and_rm() 返回 sample (status=ABORTED)
    ↓
generate_and_rm_group() 返回 group
    ↓
_make_done_cb: _has_aborted → True → data_buffer.add_samples() [requeue]
    ↓
下一轮 worker 取出 → generate_with_tau.generate() → 清理 rollout 状态 → 重新初始化 tau-bench 环境 → 从头生成
```

注意：tau-bench 环境**不可序列化/不可恢复**（会话状态在 tau-bench 进程中），所以 requeue 后只能从头重跑。这是可接受的——abort 的频率由 `update-weights-interval` 控制，不会太频繁。

---

## 7. 风险与注意事项

| 风险 | 缓解 |
|------|------|
| **logprobs 改造引入 bug** | 保留 `_get_token_delta` 作为 fallback；logprobs 只在 LLM 生成步骤使用，工具/环境 token 仍用 delta |
| **tau-bench 环境 abort 后不可恢复** | 可接受；abort 频率低（每 N 步权重同步一次） |
| **Gemini API 并发限制** | 已有 `sglang-server-concurrency` 参数控制，必要时降低 |
| **14B 模型 rollout 吞吐** | 4 卡 rollout TP=2 → 2 个 engine，`sglang-server-concurrency=128`，对 tau-bench 短轨迹（max 1024 tokens）足够 |
| **eval 在 fully async 下不支持** | `generate_rollout_fully_async` 对 `evaluation=True` 直接抛异常；暂时去掉 eval 或单独跑 |
| **Qwen3-14B 的 Instruct 版本** | 需要用 `Qwen3-14B-Instruct`（带 chat template），参考 `qwen3-4B-Instruct-2507.sh` 的 `--rotary-base 5000000` 覆盖 |

---

## 8. 实施顺序

1. 修复 `trainable_agents.py` 的 logprobs + `generate_with_tau.py` 的 `InteractionResult`
2. 写启动脚本
3. 在 8xH200 上跑通一轮（`--num-rollout 2`），验证无 crash
4. 验证 logprobs 正确 → 检查训练 loss 正常
5. 调整 `update-weights-interval` 和 `staleness-threshold` 到目标值
6. 正式训练
