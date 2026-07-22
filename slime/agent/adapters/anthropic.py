"""Anthropic Messages adapter for agent rollouts.

Exposes /v1/messages and /v1/messages/count_tokens. Each Anthropic message
history is rendered with the served model's chat template, sent to sglang
/generate as input_ids, and fed into a shared TrajectoryManager keyed by session
id. finish_session(sid) drains a session's trajectory into a list of Sample.

The per-sid tree inside TrajectoryManager handles sub-agent and compaction
patterns automatically: any divergence in the prompt prefix forks into a new
leaf, so we do not track explicit chains here.

This module mirrors slime.agent.adapters.openai; the section layout (adapter
class -> translation -> reply building -> request framing) is shared between
them. See BaseAdapter for the hooks to fill.
"""

from __future__ import annotations

import json
import logging
import secrets
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import (
    BaseAdapter,
    Reply,
    flatten_content,
    manager_finish_reason,
    sid_from_bearer,
    tool_call_dict,
)
from slime.agent.parsing import ParsedModelOutput

logger = logging.getLogger(__name__)


class AnthropicAdapter(BaseAdapter):
    """Anthropic Messages-compatible HTTP adapter: wire translation and reply
    framing only; the turn machinery is inherited from BaseAdapter."""

    logger = logger
    log_prefix = "anthropic_adapter"
    max_token_keys = ("max_tokens",)
    stop_keys = ("stop_sequences",)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Per-session code accumulator: replays past function definitions so
        # the model can define a function in one turn and call it in later turns.
        self._code_cache: dict[str, list[str]] = {}

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/messages", self._run_turn)
        app.router.add_post("/v1/messages/count_tokens", _count_tokens)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return _request_session_id(request)

    def _preprocess_body(self, body: dict) -> None:
        _fold_mid_list_system_into_user(body)

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        translated = _translate_messages(body.get("messages") or [], body.get("system"))
        tools_schema = _tools_to_chat_tools(body.get("tools"))
        return translated, tools_schema

    async def finish_session(self, sid, *, base_sample, reward=0.0, extra_metadata=None, wait_timeout=5.0) -> list:
        samples = await super().finish_session(sid, base_sample=base_sample, reward=reward, extra_metadata=extra_metadata, wait_timeout=wait_timeout)
        self._code_cache.pop(sid, None)
        return samples

    async def drop_session(self, sid, *, wait_timeout=5.0) -> None:
        await super().drop_session(sid, wait_timeout=wait_timeout)
        self._code_cache.pop(sid, None)

    def _build_reply(self, parsed, raw_finish, translated, tools_schema, sid="") -> Reply:
        blocks, stop_reason, manager_message = _build_reply_parts(
            parsed, raw_finish, code_cache=self._code_cache, sid=sid,
        )
        return Reply(
            manager_message=manager_message,
            finish_reason=manager_finish_reason(parsed.tool_uses, raw_finish),
            wire=(blocks, stop_reason),
        )

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        blocks, stop_reason = reply.wire
        if stream:
            return await _render_stream(request, blocks, stop_reason, in_tok, out_tok)
        return web.json_response(_render_response(body, blocks, stop_reason, in_tok, out_tok))


# --- Translation (Anthropic wire -> chat-template messages) ---


def _translate_messages(msgs: list[dict], system: Any) -> list[dict]:
    """Anthropic messages + system -> chat-template messages. Pure function."""
    translated: list[dict] = []
    if system:
        translated.append({"role": "system", "content": flatten_content(system)})
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": flatten_content(content)}]
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    translated.append({"role": "tool", "content": flatten_content(b.get("content"))})
                elif isinstance(b, dict) and b.get("type") == "text":
                    translated.append({"role": "user", "content": b.get("text", "")})
                else:
                    translated.append({"role": "user", "content": flatten_content(b)})
        elif role == "assistant":
            texts, thinkings, tcs = [], [], []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": flatten_content(content)}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "thinking":
                    thinkings.append(b.get("thinking", ""))
                elif b.get("type") == "tool_use":
                    # drop the wire-only id; tool_call_dict keeps arguments a dict
                    tcs.append(tool_call_dict(b.get("name", "tool"), b.get("input")))
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                mo["reasoning_content"] = "".join(thinkings)
            if tcs:
                mo["tool_calls"] = tcs
            translated.append(mo)
        elif role == "system":
            translated.append({"role": "system", "content": flatten_content(content)})
    return translated


def _tools_to_chat_tools(anth_tools: list[dict] | None) -> list[dict] | None:
    """Convert Anthropic tools to tokenizer chat-template tool schema."""
    if not anth_tools:
        return None
    ts: list[dict] = []
    for t in anth_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        ts.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return ts or None


# --- Reply building: parsed output -> Anthropic blocks + manager_message ---

import re as _re

# Injected into Bash tool calls when the model uses DataMind-specific functions
_GET_DB_INFO_DEF = """
def get_db_info():
    \"\"\"Display database schema.\"\"\"
    import sqlite3, os
    for f in sorted(os.listdir('data/files')):
        if f.endswith('.sqlite'):
            conn = sqlite3.connect(os.path.join('data/files', f))
            for row in conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"):
                print(row[0])
            conn.close()
""".strip()

_EXECUTE_SQL_DEF = """
def execute_sql(sql, output_path):
    \"\"\"Execute SQL and save results to CSV.\"\"\"
    import sqlite3, os, csv
    for f in sorted(os.listdir('data/files')):
        if f.endswith('.sqlite'):
            conn = sqlite3.connect(os.path.join('data/files', f))
            cur = conn.execute(sql)
            rows = cur.fetchall()
            if rows:
                cols = [d[0] for d in cur.description]
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                with open(output_path, 'w', newline='') as fp:
                    w = csv.writer(fp)
                    w.writerow(cols)
                    w.writerows(rows)
                print(f'Output saved to: {output_path}')
                for row in rows:
                    print(','.join(str(x) for x in row))
            else:
                print(f'Output saved to: {output_path}')
                print('(empty result)')
            conn.close()
            break
""".strip()

_MARKDOWN_CODE_RE = _re.compile(r'```(?:bash|python|sql|sh)?\s*\n(.*?)```', _re.DOTALL)
_EOS_CLEANUP_RE = _re.compile(r'<\|\s*im_end\s*\|>|<\|endoftext\|>')


_HEREDOC_RE = _re.compile(r"python3?\s*<<\s*['\"]?PYEOF['\"]?\s*\n(.*?)PYEOF", _re.DOTALL)
# Intercept echo '{"answer":"..."}' > answer.json — fragile in shell, convert to python3 heredoc
_ANSWER_JSON_RE = _re.compile(
    r"""echo\s+(['"])\s*\{.*?"answer"\s*:\s*"(.+?)"\s*[,}].*?\1\s*>\s*answer\.json""",
    _re.DOTALL,
)


def _extract_python_code(code: str) -> str:
    """Extract just the Python expression from a shell command.

    Handles ``python3 -c "..."`` and heredoc (``python3 << 'PYEOF' ... PYEOF``).
    Plain Python passes through unchanged.
    """
    code = code.strip()
    # python3 << 'PYEOF'\n...\nPYEOF
    m = _HEREDOC_RE.search(code)
    if m:
        return m.group(1).strip()
    # python3 -c "..." or python3 -c '...'
    m = _re.match(r'^python3?\s+-c\s+["\'](.+?)["\']$', code, _re.DOTALL)
    if m:
        return m.group(1).strip()
    return code


def _wrap_heredoc(code: str) -> str:
    """Wrap Python code in a bash heredoc invocation."""
    code = code.strip()
    if _re.search(r'<<\s*["\']?PYEOF', code):
        return code  # already a heredoc
    return "python3 << 'PYEOF'\n" + code + '\nPYEOF'


def _inject_functions(code: str) -> str:
    """Prepend DataMind function definitions only if the model did not already
    include them (SFT v3+ teaches self-contained definitions)."""
    parts = []
    if 'get_db_info()' in code and 'def get_db_info()' not in code:
        parts.append(_GET_DB_INFO_DEF)
    if 'execute_sql(' in code and 'def execute_sql' not in code:
        parts.append(_EXECUTE_SQL_DEF)
    if not parts:
        return code
    return '\n\n'.join(parts) + '\n\n' + code


def _build_reply_parts(
    parsed: ParsedModelOutput,
    finish: str,
    *,
    code_cache: dict[str, list[str]] | None = None,
    sid: str = "",
) -> tuple[list[dict], str, dict[str, Any]]:
    """Return (anthropic blocks, wire stop_reason, manager_message).

    The tool_calls inside manager_message use canonical args (tool_call_dict) so
    this assistant turn compares equal (dict equality) to the same turn replayed
    as history on the next request.
    """
    blocks: list[dict] = []
    if parsed.reasoning:
        blocks.append({"type": "thinking", "thinking": parsed.reasoning})
    if parsed.text:
        # Clean leaked EOS tokens from the visible text
        clean_text = _EOS_CLEANUP_RE.sub('', parsed.text)
        if clean_text.strip():
            blocks.append({"type": "text", "text": clean_text})

    manager_tcs: list[dict] = []
    for tu in parsed.tool_uses:
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
        # tu_id is wire-only; tool_call_dict drops it so the leaf matches its echo
        manager_tcs.append(tool_call_dict(tu["name"], tu.get("input")))

    # If the parser didn't find any tool_use but the model output contains
    # markdown code blocks (common after DataMind SFT), convert them to Bash
    # tool_use blocks so Claude Code executes them and continues the loop.
    if not parsed.tool_uses and parsed.text:
        code_blocks = _MARKDOWN_CODE_RE.findall(parsed.text)
        if code_blocks:
            for code in code_blocks:
                raw_code = code.strip()
                # Intercept echo '{"answer":"..."}' > answer.json — convert to python3 heredoc
                m = _ANSWER_JSON_RE.search(raw_code)
                if m:
                    answer_text = m.group(2)
                    import json as _json
                    answer_json = _json.dumps({"answer": answer_text, "reasoning": "done"}, ensure_ascii=False)
                    raw_code = f"python3 << 'PYEOF'\nimport json\njson.dump({answer_json}, open('answer.json', 'w'), ensure_ascii=False)\nPYEOF"
                current = _extract_python_code(raw_code)
                # Replay past definitions so functions persist across tool calls
                to_run = current
                if code_cache is not None and sid:
                    history = code_cache.get(sid, [])
                    if history:
                        to_run = "\n".join(history) + "\n" + current
                to_run = _inject_functions(to_run)
                command = _wrap_heredoc(to_run)
                # Cache the current code for future turns
                if code_cache is not None and sid:
                    code_cache.setdefault(sid, []).append(current)
                tu_id = f"toolu_{secrets.token_hex(8)}"
                blocks.append({"type": "tool_use", "id": tu_id, "name": "Bash", "input": {"command": command}})
                manager_tcs.append(tool_call_dict("Bash", {"command": command}))

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if manager_tcs:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    manager_message: dict[str, Any] = {"role": "assistant", "content": parsed.text or ""}
    if parsed.reasoning:
        manager_message["reasoning_content"] = parsed.reasoning
    if manager_tcs:
        manager_message["tool_calls"] = manager_tcs

    return blocks, stop_reason, manager_message


# --- Request framing: session id + wire response/stream rendering ---


def _request_session_id(request: web.Request) -> str:
    # Anthropic auth lands in Authorization: Bearer or X-Api-Key; the Messages
    # body carries no sid hint. Bearer wins when both are present.
    return sid_from_bearer(request) or (request.headers.get("X-Api-Key") or "").strip() or "default"


def _render_response(body: dict, blocks: list[dict], stop_reason: str, in_tok: int, out_tok: int) -> dict:
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "slime-actor"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


async def _render_stream(request, blocks, stop_reason, in_tok, out_tok) -> web.StreamResponse:
    """Stream blocks back as an Anthropic Messages SSE response: message_start,
    (content_block_start, content_block_delta, content_block_stop)*N,
    message_delta, message_stop."""
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)

    ms_data = {
        "type": "message_start",
        "message": {
            "id": f"msg_{secrets.token_hex(12)}",
            "type": "message",
            "role": "assistant",
            "model": "slime-actor",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": 0},
        },
    }
    await out.write(f"event: message_start\ndata: {json.dumps(ms_data, ensure_ascii=False)}\n\n".encode())

    for idx, block in enumerate(blocks):
        bt = block["type"]
        if bt == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        elif bt == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:  # tool_use
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"], ensure_ascii=False),
            }

        cbs_data = {"type": "content_block_start", "index": idx, "content_block": start}
        await out.write(f"event: content_block_start\ndata: {json.dumps(cbs_data, ensure_ascii=False)}\n\n".encode())

        cbd_data = {"type": "content_block_delta", "index": idx, "delta": delta}
        await out.write(f"event: content_block_delta\ndata: {json.dumps(cbd_data, ensure_ascii=False)}\n\n".encode())

        cbe_data = {"type": "content_block_stop", "index": idx}
        await out.write(f"event: content_block_stop\ndata: {json.dumps(cbe_data, ensure_ascii=False)}\n\n".encode())

    md_data = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }
    await out.write(f"event: message_delta\ndata: {json.dumps(md_data, ensure_ascii=False)}\n\n".encode())

    mst_data = {"type": "message_stop"}
    await out.write(f"event: message_stop\ndata: {json.dumps(mst_data, ensure_ascii=False)}\n\n".encode())

    return out


# count_tokens runs every turn but the client uses it only as a hint, not a
# hard budget, so returning 0 is fine.
async def _count_tokens(request: web.Request) -> web.Response:
    await request.read()
    return web.json_response({"input_tokens": 0})


# --- Anthropic-specific quirks: mid-list system folding ---


_MID_SYSTEM_WRAP_PREFIX = "<system-reminder>\n"
_MID_SYSTEM_WRAP_SUFFIX = "\n</system-reminder>\n"


def _fold_mid_list_system_into_user(body_obj: dict) -> bool:
    """Fold non-leading role:system messages into a neighbouring user message as
    a <system-reminder> text block. Mutates body_obj in place; returns True iff
    any fold happened.

    Some clients insert a system message in the middle of the message list, but
    many chat templates reject any system message past index 0. Attaching the
    wrapped reminder to the preceding user message (or the next one, if there is
    no prior user message) keeps the history acceptable to the template.
    """
    msgs = body_obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False

    system_idx = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "system" and i > 0]
    if not system_idx:
        return False

    def _promote_to_list(msg: dict) -> list:
        c = msg.get("content")
        if isinstance(c, list):
            return c
        msg["content"] = [{"type": "text", "text": c if isinstance(c, str) else ""}]
        return msg["content"]

    def _wrap(text: str) -> dict:
        return {
            "type": "text",
            "text": _MID_SYSTEM_WRAP_PREFIX + text + _MID_SYSTEM_WRAP_SUFFIX,
        }

    changed = False
    TOMBSTONE: dict = {"__folded__": True}
    for i in system_idx:
        sys_msg = msgs[i]
        wrapped = _wrap(flatten_content(sys_msg.get("content")))
        target = None
        for j in range(i - 1, -1, -1):
            cand = msgs[j]
            if isinstance(cand, dict) and cand.get("role") == "user":
                target = cand
                _promote_to_list(target).append(wrapped)
                break
        if target is None:
            for j in range(i + 1, len(msgs)):
                cand = msgs[j]
                if isinstance(cand, dict) and cand.get("role") == "user":
                    target = cand
                    _promote_to_list(target).insert(0, wrapped)
                    break
        if target is None:
            msgs[i] = {"role": "user", "content": [wrapped]}
            changed = True
            continue
        msgs[i] = TOMBSTONE
        changed = True

    if changed:
        body_obj["messages"] = [m for m in msgs if m is not TOMBSTONE]
    return changed
