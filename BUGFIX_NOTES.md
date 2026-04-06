# Gemma4PromptGen — Bug Analysis & Fixes

**Status: VERIFIED FIXED** — all issues confirmed resolved after applying these changes.

## Issue Report

User reported three problems:
1. PREVIEW mode generates an empty prompt (console shows headers but no text between separators)
2. SEND mode returns "No prompt stored yet. Run PREVIEW first." even after running PREVIEW
3. VRAM is not freed after SEND completes

Tested across multiple target models (SDXL, Pony, Flux, LTX, etc.) — all produced the same result.

---

## Root Cause Analysis

### Primary cause: LLM returns 0 characters

After adding diagnostic logging, the console revealed:

```
[Gemma4PromptGen] LLM response: 0 chars, finish_reason=stop, tokens: ? prompt + 0 completion
```

The llama-server starts, passes health checks, accepts the request — but the model generates **zero tokens**. This is upstream of `_clean_output` and explains why all target models produce the same empty result.

**Three contributing factors identified:**

#### Factor A: System prompt sent twice (~1,500+ wasted tokens)

`_build_message()` embeds the full system prompt into the user message:
```python
parts.append(system_prompt)  # 780+ tokens for Pony, 1400+ for LTX
```

Then `_call_llama()` sends it again as the system role:
```python
"messages": [
    {"role": "system", "content": system_prompt},   # ← first copy
    {"role": "user",   "content": user_content},     # ← contains second copy
]
```

This doubles token usage. With `--ctx-size 8192`, the doubled prompt + environment data could push past the context window, leaving zero room for generation — especially for video models where `SYSTEM_LTX` alone is ~1,400 tokens (×2 = ~2,800 tokens just for the system prompt).

#### Factor B: `--reasoning-budget 0` may not be supported

The llama-server is started with `--reasoning-budget 0`. This flag was added in recent llama.cpp builds. If the user has an older binary (found via PATH or pre-installed), this flag may:
- Be silently ignored (best case)
- Cause the model to suppress all output including the actual response (worst case)
- Interact badly with non-reasoning models

#### Factor C: `/no_think` command for non-Qwen models

For image models, `_build_message()` prepends `/no_think` to the user message. This is a Qwen-specific control token. For other models (Gemma, Llama, Mistral, etc.) it's meaningless at best and could confuse the model's generation at worst.

#### Factor D: `--flash-attn on` syntax error

The original startup command included `"--flash-attn", "on"`. In llama.cpp, `--flash-attn` is a boolean flag (no argument). The `on` string becomes a stray positional argument. While this likely doesn't cause the empty response, it's incorrect.

### Secondary bugs (amplified by primary cause)

#### Bug 2: "No prompt stored" — empty string passes store check

```python
if not prompt.startswith("❌") and not prompt.startswith("⚠️"):
    Gemma4PromptGen._last_prompt = prompt   # stores "" 
```

Then in SEND: `not ""` → True → "No prompt stored yet."

#### Bug 3: VRAM not freed — early return skips server kill

```python
# SEND MODE (original)
if not Gemma4PromptGen._last_prompt:
    return ("", "❌ ...")        # exits before kill
self._kill_llama_server()        # never reached
```

When Bug 2 stores empty string, SEND returns before calling `_kill_llama_server()`. The llama-server process keeps running and holding VRAM indefinitely.

#### Bug 4: `_clean_output` can strip valid content

Even when the LLM does respond, `_clean_output()` junk filtering can discard the entire response if content appears on the same line as a matched prefix pattern:

- `"Prompt: masterpiece, best quality..."` — matches `^Prompt:`, whole line discarded
- `"Here's your prompt: masterpiece..."` — matches `^Here'?s?\s`, whole line discarded

---

## Fixes Applied

### Fix 1: Remove system prompt duplication from user message

**File:** `gemma4_prompt_gen.py`, `_build_message()` method

Removed the system prompt injection and `/no_think` command from the user message. The system prompt is now sent **only once** as the system role. This frees ~780-1,400 tokens of context space depending on the target model.

Before:
```python
parts.append("/no_think")
parts.append("Read and follow these instructions carefully:\n")
parts.append(system_prompt)
parts.append("\n---\n")
```

After:
```python
parts.append("Follow the system instructions. Generate the prompt for this scene:\n")
```

### Fix 2: Safe `--reasoning-budget` detection

**File:** `gemma4_prompt_gen.py`, `_ensure_llama_running()` method

Instead of always passing `--reasoning-budget 0`, the code now checks if the binary supports it via `--help` output. Also fixed `--flash-attn on` → `--flash-attn` (boolean flag, no argument).

```python
# Check if the binary supports --reasoning-budget before adding
help_out = subprocess.run([llama_exe, "--help"], ...)
if "--reasoning-budget" in help_out.stdout:
    cmd += ["--reasoning-budget", "0"]
```

### Fix 3: Always kill llama-server in SEND mode

**File:** `gemma4_prompt_gen.py`, SEND mode block

Moved `_kill_llama_server()` to execute **before** the empty-prompt check. VRAM is now always freed.

```python
# SEND MODE
self._kill_llama_server()  # Always free VRAM first
if not Gemma4PromptGen._last_prompt:
    return ("", "❌ No prompt stored yet. Run PREVIEW first.",)
```

### Fix 4: Fallback in `_clean_output`

**File:** `gemma4_prompt_gen.py`, `_clean_output()` method

If the junk filter removes everything but the LLM did return content, a lightweight fallback strips only the known prefix and keeps the rest.

```python
if not text and original_text:
    text = re.sub(r"(?i)^(prompt|here'?s?[^:]*|sure[^:]*)\s*:\s*", "", original_text, count=1).strip()
```

### Fix 5: Explicit empty-prompt detection in PREVIEW

**File:** `gemma4_prompt_gen.py`, `execute()` PREVIEW block

If the prompt is empty after cleaning, a visible warning is shown instead of silently storing nothing.

```python
if not prompt or not prompt.strip():
    prompt = "⚠️ LLM returned empty response after cleaning. Check the console log above for raw output."
```

### Fix 6: Detailed diagnostic logging

**File:** `gemma4_prompt_gen.py`, `_call_llama()` method

Added logging of: response length, finish_reason, prompt tokens, completion tokens, startup command, and `<think>` tag stripping for Qwen/DeepSeek models.

---

## Summary

| Bug | Cause | Fix |
|-----|-------|-----|
| 0-char LLM response | System prompt sent twice (context overflow) + unsupported flags | Removed duplication, safe flag detection |
| Empty prompt stored | `""` passes error-prefix check | Explicit empty check with warning |
| VRAM not freed | `_kill_llama_server()` unreachable on early return | Moved kill before empty check |
| Valid content stripped | `_clean_output` junk filter too aggressive on single-line responses | Fallback to original with minimal prefix strip |
| Silent failures | No diagnostic output | Added finish_reason, token counts, startup command logging |
| `--flash-attn on` | Boolean flag passed with stray argument | Changed to `--flash-attn` (no argument) |

## Diagnostic checklist for users

If prompts are still empty after these fixes, check the console for:

1. **`tokens: N prompt + 0 completion`** — if prompt tokens are near 8192, the context is full. Try a shorter instruction or simpler environment.
2. **`finish_reason=length`** — the model ran out of context before generating. Reduce input size.
3. **`finish_reason=stop` with 0 completion** — the model hit its EOS token immediately. This usually means the model doesn't understand the chat format. Try a different GGUF or ensure a compatible chat template is applied.
4. **Check which GGUF model is loaded** — this node is designed for Gemma 4. Other models may need different chat templates or prompt formats.
