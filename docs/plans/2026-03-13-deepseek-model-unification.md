# DeepSeek Model Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose only `deepseek-v3.2-exp` in the shared model list while keeping legacy DeepSeek request aliases working and enabling thinking by default for the unified DeepSeek family.

**Architecture:** Keep `/v1/models` as the shared Anthropic/OpenAI listing source, but collapse public DeepSeek entries to one item there. Treat both the new public alias and legacy aliases as one compatibility family in Anthropic and OpenAI request conversion so downstream callers see one model while old clients keep working.

**Tech Stack:** Rust, Axum, Serde, existing Anthropic/OpenAI request converters, module-local Rust tests.

---

### Task 1: Lock the New Public Contract with Tests

**Files:**
- Modify: `src/anthropic/handlers.rs`
- Modify: `src/anthropic/converter.rs`
- Modify: `src/openai/request_converter.rs`
- Modify: `src/openai/handlers.rs`

**Step 1: Write the failing tests**

Add tests that assert:

- `GET /v1/models` only exposes `deepseek-v3.2-exp`
- Anthropic model mapping accepts the new public alias and legacy aliases
- OpenAI request conversion accepts the new public alias and legacy aliases
- The unified DeepSeek family defaults to thinking mode

**Step 2: Run test to verify it fails**

Run: `cargo test deepseek -- --nocapture`

Expected: FAIL because the current code still exposes multiple DeepSeek IDs and does not recognize `deepseek-v3.2-exp`.

### Task 2: Collapse Public Model Exposure

**Files:**
- Modify: `src/anthropic/handlers.rs`

**Step 1: Write minimal implementation**

Replace the multiple public DeepSeek entries in `available_models()` with one entry for `deepseek-v3.2-exp`, and keep the shared `/v1/models` route untouched.

**Step 2: Run focused tests**

Run: `cargo test models_include_deepseek -- --nocapture`

Expected: PASS with only the unified public alias exposed.

### Task 3: Unify Request Alias Mapping

**Files:**
- Modify: `src/anthropic/converter.rs`
- Modify: `src/anthropic/handlers.rs`
- Modify: `src/openai/request_converter.rs`
- Modify: `src/openai/handlers.rs`

**Step 1: Write minimal implementation**

Treat `deepseek-v3.2-exp` and the legacy aliases as one DeepSeek compatibility family in both protocol paths, and make that family enable thinking by default.

**Step 2: Run focused tests**

Run: `cargo test openai::request_converter -- --nocapture`

Run: `cargo test anthropic::handlers::tests::test_available_models_include_deepseek -- --nocapture`

Expected: PASS for alias compatibility and default thinking behavior.

### Task 4: Verify End-to-End Regression Surface

**Files:**
- Modify: `README.md` if the public DeepSeek example still shows an old alias

**Step 1: Run regression checks**

Run: `cargo test deepseek -- --nocapture`

Run: `cargo test models_include_deepseek -- --nocapture`

Expected: PASS with no remaining references to public old aliases in the tested model list surface.
