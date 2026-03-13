# DeepSeek Model Unification Design

**Date:** 2026-03-13
**Status:** Approved

## Goal

Unify the OpenAI-compatible DeepSeek exposure so downstream clients only see `deepseek-v3.2-exp`, while preserving compatibility for existing requests that still send legacy DeepSeek aliases.

## Decisions

- Public model listing at `GET /v1/models` exposes only `deepseek-v3.2-exp`.
- Legacy request names remain accepted:
  - `deepseek-chat`
  - `deepseek-reasoner`
  - `deepseek-3-2`
  - `deepseek-3-2-thinking`
- The unified DeepSeek family defaults to thinking mode.
- Anthropic-compatible and OpenAI-compatible paths both accept the new public alias so the shared model list is truthful.

## Scope

- Update shared model listing in `src/anthropic/handlers.rs`.
- Update Anthropic model alias mapping in `src/anthropic/converter.rs`.
- Update OpenAI DeepSeek alias mapping and reasoning detection in `src/openai/request_converter.rs` and `src/openai/handlers.rs`.
- Refresh targeted tests that assert model exposure and DeepSeek request conversion behavior.

## Non-Goals

- Do not change non-DeepSeek model names.
- Do not add new endpoints.
- Do not remove backward compatibility for existing DeepSeek request names.
