# OpenAI Stream Reasoning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restore structured `reasoning_content` in OpenAI-compatible streaming without bringing back the slow buffering behavior that delayed plain assistant text.

**Architecture:** Keep the direct Kiro-to-OpenAI streaming path, but teach the OpenAI stream converter to parse `<thinking>` and `</thinking>` tags incrementally. Normal text should flush immediately unless the tail could be a split tag boundary; reasoning text should stream through `reasoning_content` with the same minimal-boundary buffering rule.

**Tech Stack:** Rust, Axum, futures stream unfolding, existing Kiro event decoder, module-local Rust tests.

---

### Task 1: Lock the Regression with Tests

**Files:**
- Modify: `src/openai/handlers.rs`
- Modify: `src/openai/stream.rs`

**Step 1: Write the failing tests**

Add tests that assert:

- small non-thinking chunks still forward immediately in streaming mode
- thinking chunks produce `reasoning_content` deltas instead of raw `<thinking>` text
- answer text resumes as `content` after `</thinking>`

**Step 2: Run test to verify it fails**

Run: `cargo test openai::handlers::tests::thinking_stream_preserves_reasoning_content_and_text_content -- --nocapture`

Expected: FAIL because the current OpenAI streaming path emits raw assistant content and does not restore structured reasoning deltas.

### Task 2: Reintroduce Reasoning Parsing Without Reintroducing Slow Buffering

**Files:**
- Modify: `src/openai/handlers.rs`
- Modify: `src/openai/stream.rs`

**Step 1: Write minimal implementation**

- pass explicit `thinking_enabled` into the OpenAI stream converter
- add a minimal incremental thinking parser in `OpenAiStreamConverter`
- buffer only the suffix that may be a split `<thinking>` or `</thinking>\n\n` boundary
- flush pending buffered assistant content before tool calls and before stream finish

**Step 2: Run focused tests**

Run: `cargo test openai::stream -- --nocapture`

Run: `cargo test openai::handlers::tests::thinking_stream_preserves_reasoning_content_and_text_content -- --nocapture`

Expected: PASS with separate `reasoning_content` and `content` chunks.

### Task 3: Verify the Full Regression Surface

**Files:**
- Modify: none unless a test reveals missing edge handling

**Step 1: Run full verification**

Run: `cargo test`

Expected: PASS with no OpenAI streaming regressions.

### Task 4: Publish the Fix

**Files:**
- Modify: none

**Step 1: Commit**

```bash
git add docs/plans/2026-03-13-openai-stream-reasoning-design.md docs/plans/2026-03-13-openai-stream-reasoning.md src/openai/handlers.rs src/openai/stream.rs
git commit -m "fix: restore openai streaming reasoning deltas"
```

**Step 2: Push**

```bash
git push origin main
```
