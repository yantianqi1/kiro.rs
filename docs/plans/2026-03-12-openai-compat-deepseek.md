# OpenAI-Compatible DeepSeek Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a native OpenAI-compatible `/v1/chat/completions` endpoint that reuses the current Kiro transport and migrates the Kiro OpenAI translator behavior needed for DeepSeek/OpenAI compatibility from `CLIProxyAPIPlus`.

**Architecture:** Introduce a dedicated `src/openai/` protocol layer that translates OpenAI Chat Completions directly to `KiroRequest`, then translates Kiro responses and streams back into OpenAI format. Keep the existing Anthropic layer intact and preserve the current Kiro IDE-style upstream transport by continuing to use `KiroProvider` and shared `AppState`.

**Tech Stack:** Rust, Axum, Serde, Reqwest, existing Kiro request/event models, unit tests in Rust source modules.

---

### Task 1: Scaffold OpenAI Module and Routing

**Files:**
- Create: `src/openai/mod.rs`
- Create: `src/openai/router.rs`
- Create: `src/openai/types.rs`
- Modify: `src/main.rs`
- Modify: `src/anthropic/mod.rs`
- Test: `src/openai/router.rs`

**Step 1: Write the failing test**

Add a router-level test in `src/openai/router.rs` that constructs a minimal app with shared `AppState` and asserts:

- `POST /v1/chat/completions` is routed
- unauthenticated requests return `401`

Use an empty stub provider path if needed; the test only needs to prove routing and auth wiring.

**Step 2: Run test to verify it fails**

Run: `cargo test openai::router -- --nocapture`

Expected: FAIL because the `openai` module and route do not exist.

**Step 3: Write minimal implementation**

Implement:

- `src/openai/mod.rs` module export
- `src/openai/router.rs` with a router builder using `anthropic::middleware::AppState`
- top-level app composition in `src/main.rs` to add `POST /v1/chat/completions`

Do not implement real handler logic yet; return a placeholder `501` or minimal stub to make the route compile.

**Step 4: Run test to verify it passes**

Run: `cargo test openai::router -- --nocapture`

Expected: PASS for route/auth coverage.

**Step 5: Commit**

```bash
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek add src/openai/mod.rs src/openai/router.rs src/openai/types.rs src/main.rs src/anthropic/mod.rs
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek commit -m "feat: scaffold openai chat completions routing"
```

### Task 2: Add Kiro InferenceConfig Support

**Files:**
- Modify: `src/kiro/model/requests/kiro.rs`
- Test: `src/kiro/model/requests/kiro.rs`

**Step 1: Write the failing test**

Add serialization tests for a `KiroRequest` carrying:

- `maxTokens`
- `temperature`
- `topP`

Verify the serialized JSON contains `inferenceConfig` only when at least one value is present.

**Step 2: Run test to verify it fails**

Run: `cargo test kiro_request_inference_config -- --nocapture`

Expected: FAIL because `KiroRequest` has no `inferenceConfig`.

**Step 3: Write minimal implementation**

Add:

- `InferenceConfig` struct
- optional `inference_config` field on `KiroRequest`

Ensure serde output uses `camelCase`.

**Step 4: Run test to verify it passes**

Run: `cargo test kiro_request_inference_config -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek add src/kiro/model/requests/kiro.rs
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek commit -m "feat: add kiro inference config support"
```

### Task 3: Implement OpenAI Request Conversion

**Files:**
- Create: `src/openai/request_converter.rs`
- Modify: `src/openai/mod.rs`
- Modify: `src/openai/types.rs`
- Test: `src/openai/request_converter.rs`

**Step 1: Write the failing test**

Add request converter tests covering:

- DeepSeek plain model -> Kiro `modelId`
- DeepSeek thinking request using `reasoning_effort`
- `max_tokens`, `temperature`, `top_p` -> `inferenceConfig`
- `assistant.tool_calls` -> Kiro `toolUses`
- `tool` role messages -> `toolResults`
- current user message gets pending tool results
- orphaned tool results are dropped

Use payloads modeled after `CLIProxyAPIPlus/internal/translator/kiro/openai/kiro_openai_request_test.go`.

**Step 2: Run test to verify it fails**

Run: `cargo test openai::request_converter -- --nocapture`

Expected: FAIL because the converter does not exist.

**Step 3: Write minimal implementation**

Implement a Rust converter that builds `KiroRequest` directly from OpenAI Chat Completions input, including:

- model mapping and DeepSeek aliases
- `origin = "AI_EDITOR"`
- OpenAI messages/history conversion
- tool/tool result handling
- thinking detection from `reasoning_effort` and model hints
- `inferenceConfig`

Keep scope limited to behavior required by the approved design.

**Step 4: Run test to verify it passes**

Run: `cargo test openai::request_converter -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek add src/openai/request_converter.rs src/openai/mod.rs src/openai/types.rs
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek commit -m "feat: convert openai chat requests to kiro payloads"
```

### Task 4: Implement OpenAI Non-Stream Response Conversion

**Files:**
- Create: `src/openai/response_converter.rs`
- Modify: `src/openai/mod.rs`
- Test: `src/openai/response_converter.rs`

**Step 1: Write the failing test**

Add non-stream conversion tests for:

- text-only assistant output
- thinking block -> `reasoning_content`
- tool use -> OpenAI `tool_calls`
- `stop_reason` mapping to `finish_reason`
- usage mapping to `prompt_tokens`, `completion_tokens`, `total_tokens`

**Step 2: Run test to verify it fails**

Run: `cargo test openai::response_converter -- --nocapture`

Expected: FAIL because the converter does not exist.

**Step 3: Write minimal implementation**

Implement Kiro/Claude-compatible response parsing into OpenAI Chat Completions JSON. Prefer pure conversion helpers with no HTTP concerns.

**Step 4: Run test to verify it passes**

Run: `cargo test openai::response_converter -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek add src/openai/response_converter.rs src/openai/mod.rs
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek commit -m "feat: translate kiro responses to openai format"
```

### Task 5: Implement OpenAI Streaming Conversion

**Files:**
- Create: `src/openai/stream.rs`
- Modify: `src/openai/mod.rs`
- Test: `src/openai/stream.rs`

**Step 1: Write the failing test**

Add streaming conversion tests that feed Claude/Kiro-style SSE events and assert OpenAI chunk output for:

- first assistant role chunk
- text deltas
- reasoning deltas
- tool call start and argument deltas
- final finish chunk
- done marker behavior

**Step 2: Run test to verify it fails**

Run: `cargo test openai::stream -- --nocapture`

Expected: FAIL because the stream converter does not exist.

**Step 3: Write minimal implementation**

Implement an OpenAI stream state machine modeled after the Go translator:

- parse event/data lines
- emit OpenAI chunk JSON
- preserve ordering
- avoid `/cc/v1/messages`-style buffering

**Step 4: Run test to verify it passes**

Run: `cargo test openai::stream -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek add src/openai/stream.rs src/openai/mod.rs
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek commit -m "feat: add openai streaming conversion for kiro events"
```

### Task 6: Implement OpenAI Chat Completions Handler

**Files:**
- Create: `src/openai/handlers.rs`
- Modify: `src/openai/router.rs`
- Modify: `src/openai/mod.rs`
- Test: `src/openai/handlers.rs`

**Step 1: Write the failing test**

Add handler tests that cover:

- non-stream request calls provider and returns OpenAI JSON
- stream request returns `text/event-stream`
- provider failure maps to OpenAI-compatible error envelope
- shared auth middleware still applies

Use a test helper that injects fake Kiro responses or minimal parseable payloads without requiring live upstream access.

**Step 2: Run test to verify it fails**

Run: `cargo test openai::handlers -- --nocapture`

Expected: FAIL because the handler does not exist.

**Step 3: Write minimal implementation**

Implement `POST /v1/chat/completions`:

- parse OpenAI request
- convert to `KiroRequest`
- send with existing `KiroProvider`
- return non-stream or stream OpenAI-compatible output
- reuse current provider error mapping where possible, but emit OpenAI-shaped errors

**Step 4: Run test to verify it passes**

Run: `cargo test openai::handlers -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek add src/openai/handlers.rs src/openai/router.rs src/openai/mod.rs
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek commit -m "feat: add openai chat completions handler"
```

### Task 7: Verify Shared Model Listing and No Anthropic Regression

**Files:**
- Modify: `src/anthropic/handlers.rs`
- Modify: `src/openai/types.rs`
- Test: `src/anthropic/handlers.rs`
- Test: `src/openai/handlers.rs`

**Step 1: Write the failing test**

Add tests asserting:

- `GET /v1/models` still returns current model IDs
- DeepSeek model entries remain present
- Anthropic `/v1/messages` tests still pass after OpenAI route wiring

**Step 2: Run test to verify it fails**

Run: `cargo test models_include_deepseek -- --nocapture`

Expected: FAIL only if the route/model integration regressed during OpenAI work.

**Step 3: Write minimal implementation**

Make only the smallest changes needed to keep shared model metadata coherent between Anthropic and OpenAI clients. Do not rename or remove existing Anthropic model IDs.

**Step 4: Run test to verify it passes**

Run: `cargo test models_include_deepseek -- --nocapture`

Expected: PASS.

**Step 5: Commit**

```bash
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek add src/anthropic/handlers.rs src/openai/types.rs src/openai/handlers.rs
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek commit -m "test: verify shared model listing and protocol compatibility"
```

### Task 8: Final Verification and Documentation Touch-Up

**Files:**
- Modify: `README.md`
- Test: `README.md`

**Step 1: Write the failing test**

Instead of a code test, add a verification checklist entry in the README updates and confirm the documented endpoints match the implemented routes.

**Step 2: Run test to verify it fails**

Run:

```bash
rg -n "/v1/chat/completions|OpenAI" README.md
```

Expected: no matching endpoint documentation yet, or outdated docs.

**Step 3: Write minimal implementation**

Update `README.md` to describe:

- OpenAI-compatible `/v1/chat/completions`
- existing Anthropic endpoints remain available
- `/cc/v1/messages` buffering distinction still applies only to Claude Code compatibility

**Step 4: Run test to verify it passes**

Run:

```bash
rg -n "/v1/chat/completions|OpenAI" README.md
cargo test openai:: -- --nocapture
```

Expected: README references present; OpenAI module tests pass.

**Step 5: Commit**

```bash
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek add README.md
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek commit -m "docs: describe openai-compatible chat completions endpoint"
```

### Task 9: Full Regression Verification

**Files:**
- Test: `src/openai/*.rs`
- Test: `src/anthropic/*.rs`
- Test: `src/kiro/model/requests/kiro.rs`

**Step 1: Write the failing test**

No new test file. This task is the final execution gate for the tests written in earlier tasks.

**Step 2: Run test to verify it fails**

Run:

```bash
cargo test openai:: -- --nocapture
```

Expected: any remaining failures must be fixed before completion.

**Step 3: Write minimal implementation**

Fix the smallest remaining issues only. Do not refactor unrelated code.

**Step 4: Run test to verify it passes**

Run:

```bash
cargo test openai:: -- --nocapture
cargo test deepseek -- --nocapture
cargo test models_include_deepseek -- --nocapture
```

Expected: PASS for the targeted OpenAI/DeepSeek coverage.

If the branch becomes buildable without external frontend artifacts, also run:

```bash
cargo test
```

**Step 5: Commit**

```bash
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek add .
git -C /Users/项目/kirors/kiro.rs/.worktrees/openai-compat-deepseek commit -m "feat: add openai-compatible deepseek path for kiro gateway"
```
