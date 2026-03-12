# OpenAI-Compatible DeepSeek Design

**Date:** 2026-03-12
**Worktree:** `feature/openai-compat-deepseek`
**Status:** Approved

## Goal

Add standard OpenAI-compatible endpoints to the current Kiro gateway so downstream clients can connect with only the project base URL and call existing models through:

- `GET /v1/models`
- `POST /v1/chat/completions`

The new OpenAI path must reproduce the Kiro OpenAI translator behavior from `CLIProxyAPIPlus` closely enough that `deepseek` requests behave like that project, while preserving the current Kiro IDE-style upstream transport and leaving Anthropic-compatible endpoints intact.

## Non-Goals

- Do not add `/v1/completions`.
- Do not add `/v1/responses`.
- Do not migrate the Go project's registry, executor, auth, or multi-provider framework.
- Do not route OpenAI requests through the existing Anthropic translator as an intermediate step.
- Do not change the semantics of `/v1/messages` or `/cc/v1/messages`.
- Do not address unrelated baseline issues such as the missing `admin-ui/dist` build artifact in this branch.

## Current State

The current Rust project is an Anthropic-compatible gateway that ultimately forwards requests to Kiro through `KiroProvider`. That upstream transport is already the part that makes requests look like Kiro IDE traffic:

- Kiro IDE-style `User-Agent` and `x-amz-user-agent`
- `x-amzn-kiro-agent-mode: vibe`
- Kiro/AWS host and auth flow
- shared token refresh, region, proxy, and retry handling

Today every external request format must first become Anthropic `messages`, then be converted into `KiroRequest`. That creates two problems for the requested feature:

1. OpenAI clients cannot connect directly because `/v1/chat/completions` does not exist.
2. DeepSeek behavior cannot match `CLIProxyAPIPlus` closely when requests are funneled through Anthropic semantics first.

## Chosen Approach

Add a separate OpenAI protocol layer that translates OpenAI Chat Completions requests directly into `KiroRequest`, then reuses the existing `KiroProvider` transport unchanged.

This keeps the fastest path:

`OpenAI Chat Completions -> KiroRequest -> KiroProvider -> upstream Kiro API`

and avoids the slower path:

`OpenAI -> Anthropic -> Kiro`

The OpenAI layer will migrate only the Kiro/OpenAI translator behavior from `CLIProxyAPIPlus`, not the surrounding framework. The Rust gateway remains the single transport implementation.

## Why This Preserves Kiro IDE Cloaking

The Kiro IDE disguise is not implemented by the Anthropic translator. It is implemented by the transport layer in `KiroProvider` and related auth/token machinery.

As long as the new OpenAI endpoint still calls:

- the same `KiroProvider`
- the same credential selection
- the same Kiro headers
- the same Kiro host/region logic

the upstream request continues to look like the current Kiro IDE-style traffic. The OpenAI layer only changes the downstream protocol and the in-memory request/response translation.

To avoid introducing CLI quota semantics, the new translator will use `origin = "AI_EDITOR"` for OpenAI requests.

## API Surface

The service will expose both compatibility layers side by side:

- `GET /v1/models`
- `POST /v1/messages`
- `POST /v1/messages/count_tokens`
- `POST /v1/chat/completions`
- `POST /cc/v1/messages`
- `POST /cc/v1/messages/count_tokens`

Format "auto-detection" is done by standard endpoint shape, not by guessing from body structure on a shared path. Downstream OpenAI clients only need the normal OpenAI base URL and will naturally use `/v1/models` and `/v1/chat/completions`.

## Module Layout

Create a new `src/openai/` module:

- `src/openai/mod.rs`
- `src/openai/router.rs`
- `src/openai/types.rs`
- `src/openai/request_converter.rs`
- `src/openai/response_converter.rs`
- `src/openai/stream.rs`

Responsibilities:

- `types.rs`
  OpenAI Chat Completions request/response types plus shared JSON helpers.
- `request_converter.rs`
  OpenAI request -> `KiroRequest`.
- `response_converter.rs`
  Kiro non-stream result -> OpenAI chat completion JSON.
- `stream.rs`
  Kiro event stream -> OpenAI chunk SSE stream.
- `router.rs`
  `/v1/chat/completions` endpoint wiring using the shared `AppState`.

Shared state will continue to reuse `anthropic::middleware::AppState` to avoid duplicating auth/provider wiring.

## Request Conversion Behavior To Migrate

The following behavior from `CLIProxyAPIPlus/internal/translator/kiro/openai` will be migrated:

### OpenAI request parsing

- `model`
- `messages`
- `stream`
- `max_tokens`
- `temperature`
- `top_p`
- `tools`
- `tool_choice`
- `response_format` only as system-hint injection where relevant
- `reasoning_effort`

### History and tool semantics

- `assistant.tool_calls` become Kiro `toolUses`
- `tool` role messages become `toolResults`
- pending tool results attach to the next user message
- current user message keeps tool results in `userInputMessageContext`
- orphaned tool results are filtered out
- adjacent/empty history edge cases get non-empty fallback content when Kiro requires it

### Thinking and DeepSeek behavior

Thinking mode is enabled when any of the migrated signals indicate it:

- `reasoning_effort` is present and not `none`
- the model name hints at thinking
- request content already contains thinking tags

The translator will preserve the same practical behavior target as the Go project:

- inject Kiro thinking tags once
- avoid duplicate thinking injection
- expose OpenAI-compatible reasoning output as `reasoning_content`

### Inference configuration

The current Rust `KiroRequest` does not carry inference settings. To match the Go translator behavior and improve DeepSeek responsiveness, the Rust request model will gain:

- `inferenceConfig.maxTokens`
- `inferenceConfig.temperature`
- `inferenceConfig.topP`

This is required to stop silently dropping user inference parameters on the OpenAI path.

## Response Conversion Behavior To Migrate

### Non-streaming

Convert the Kiro response into OpenAI Chat Completions JSON:

- `text` blocks -> assistant `content`
- `thinking` blocks -> `reasoning_content`
- `tool_use` blocks -> `tool_calls`
- Kiro/Claude-style `stop_reason` -> OpenAI `finish_reason`
- usage -> `prompt_tokens`, `completion_tokens`, `total_tokens`

### Streaming

Convert Kiro/Claude-style SSE events directly into OpenAI chunk SSE:

- `message_start` -> first assistant role chunk
- `content_block_delta.text_delta` -> text chunk
- `content_block_delta.thinking_delta` -> `reasoning_content` chunk
- `tool_use` start and JSON deltas -> tool call chunks
- `message_delta.stop_reason` -> final OpenAI finish chunk
- stream close -> `[DONE]`

The OpenAI path will be real-time streaming only. It will not adopt the `/cc/v1/messages` buffering behavior because that would reduce perceived speed.

## Models and Naming

`/v1/models` remains the shared listing endpoint. The implementation should continue exposing the project's current model IDs and add any Kiro-prefixed DeepSeek aliases needed for OpenAI clients only if the translator requires them.

The key rule is compatibility, not renaming everything. Existing Anthropic users must not lose access to current model names.

## Error Handling

OpenAI endpoints will map transport and upstream failures into OpenAI-compatible error envelopes:

- auth failure -> `authentication_error`
- malformed request -> `invalid_request_error`
- upstream Kiro failure -> `api_error`

The underlying retry/credential logic stays in `KiroProvider`.

For streaming errors:

- fail before upstream connection -> regular JSON error response
- fail after stream start -> emit a final OpenAI-compatible stream error event if feasible, otherwise close the stream cleanly after logging

## Testing Strategy

Testing will be TDD-first and cover four layers:

1. Request conversion unit tests
   - DeepSeek plain model
   - DeepSeek thinking model
   - tool calls
   - tool result history attachment
   - `reasoning_effort`
   - inference parameter propagation

2. Non-stream response conversion unit tests
   - text-only
   - tool-calls
   - reasoning content
   - finish reason mapping

3. Streaming conversion unit tests
   - first role chunk
   - text deltas
   - reasoning deltas
   - tool call start/arguments
   - final chunk and `[DONE]`

4. Router/handler integration tests
   - `POST /v1/chat/completions` non-stream
   - `POST /v1/chat/completions` stream
   - `/v1/messages` unchanged
   - `/v1/models` still usable

## Key Risks

- Kiro request shape mismatch while adding `inferenceConfig`
- finish reason mismatch between current Rust behavior and the Go translator
- duplicated model exposure in `/v1/models`
- accidental regression of Anthropic endpoints by over-sharing code

Mitigations:

- isolate OpenAI code in its own module
- keep `KiroProvider` untouched
- write converter tests before implementation
- avoid changing current Anthropic converter behavior unless required by shared request structs

## Success Criteria

The feature is successful when:

1. A downstream OpenAI-compatible client can connect using only the project base URL.
2. The client can call `GET /v1/models` and `POST /v1/chat/completions`.
3. `deepseek` requests on the OpenAI path behave like the Kiro OpenAI translator in `CLIProxyAPIPlus`.
4. The upstream transport still uses the current Kiro IDE-style request path and headers.
5. Existing Anthropic-compatible users see no protocol regression.
