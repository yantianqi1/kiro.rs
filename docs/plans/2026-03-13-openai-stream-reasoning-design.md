# OpenAI Stream Reasoning Design

**Date:** 2026-03-13
**Status:** Approved

## Goal

Keep OpenAI-compatible DeepSeek streaming as close to upstream chunk timing as possible while preserving structured reasoning output for downstream clients that expect `reasoning_content`.

## Decisions

- The OpenAI streaming path must keep emitting normal answer text as `delta.content`.
- `<thinking>...</thinking>` output from upstream must be exposed as `delta.reasoning_content`, not mixed into normal `content`.
- The streaming converter may buffer only the minimum suffix needed to detect split thinking tags across chunk boundaries.
- Tool call streaming and finish-reason handling must continue to work after the reasoning parser is added back.

## Scope

- Update `src/openai/handlers.rs` to pass explicit reasoning mode into the streaming converter.
- Update `src/openai/stream.rs` to parse thinking tags directly from Kiro assistant chunks with minimal buffering.
- Add regression tests that cover:
  - immediate forwarding of plain text chunks
  - structured reasoning deltas during streaming
  - normal text resuming after thinking ends

## Non-Goals

- Do not reintroduce the full Anthropic `StreamContext` into the OpenAI streaming path.
- Do not change non-streaming OpenAI response conversion.
- Do not change public model names or request alias mapping.
