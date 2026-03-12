use chrono::Utc;
use uuid::Uuid;

use crate::anthropic::{SseEvent, StreamContext};
use crate::kiro::model::events::Event;

use super::types::{
    OpenAiAssistantMessage, OpenAiChatCompletionResponse, OpenAiChoice, OpenAiToolCall,
    OpenAiUsage, ToolCallFunction,
};

pub fn convert_events_to_response(
    model: &str,
    input_tokens: i32,
    thinking_enabled: bool,
    events: &[Event],
) -> OpenAiChatCompletionResponse {
    let mut ctx = StreamContext::new_with_thinking(model, input_tokens, thinking_enabled);
    let mut sse_events = ctx.generate_initial_events();
    for event in events {
        sse_events.extend(ctx.process_kiro_event(event));
    }
    sse_events.extend(ctx.generate_final_events());

    convert_sse_events_to_response(model, &sse_events)
}

fn convert_sse_events_to_response(
    model: &str,
    sse_events: &[SseEvent],
) -> OpenAiChatCompletionResponse {
    let mut text_content = String::new();
    let mut reasoning_content = String::new();
    let mut tool_calls = Vec::new();
    let mut active_tool_calls = std::collections::HashMap::<i32, usize>::new();
    let mut prompt_tokens = 0;
    let mut completion_tokens = 0;
    let mut finish_reason = Some("stop".to_string());

    for event in sse_events {
        match event.event.as_str() {
            "message_start" => {
                prompt_tokens = event.data["message"]["usage"]["input_tokens"]
                    .as_i64()
                    .unwrap_or_default() as i32;
            }
            "content_block_start" => {
                if event.data["content_block"]["type"] == "tool_use" {
                    let index = event.data["index"].as_i64().unwrap_or_default() as i32;
                    let tool_call = OpenAiToolCall {
                        id: event.data["content_block"]["id"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        tool_type: "function".to_string(),
                        function: ToolCallFunction {
                            name: event.data["content_block"]["name"]
                                .as_str()
                                .unwrap_or_default()
                                .to_string(),
                            arguments: String::new(),
                        },
                    };
                    active_tool_calls.insert(index, tool_calls.len());
                    tool_calls.push(tool_call);
                }
            }
            "content_block_delta" => {
                let delta = &event.data["delta"];
                match delta["type"].as_str().unwrap_or_default() {
                    "text_delta" => {
                        text_content.push_str(delta["text"].as_str().unwrap_or_default());
                    }
                    "thinking_delta" => {
                        reasoning_content
                            .push_str(delta["thinking"].as_str().unwrap_or_default());
                    }
                    "input_json_delta" => {
                        let block_index = event.data["index"].as_i64().unwrap_or_default() as i32;
                        if let Some(tool_position) = active_tool_calls.get(&block_index) {
                            tool_calls[*tool_position]
                                .function
                                .arguments
                                .push_str(delta["partial_json"].as_str().unwrap_or_default());
                        }
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                prompt_tokens = event.data["usage"]["input_tokens"]
                    .as_i64()
                    .unwrap_or(prompt_tokens as i64) as i32;
                completion_tokens = event.data["usage"]["output_tokens"]
                    .as_i64()
                    .unwrap_or_default() as i32;
                finish_reason = map_finish_reason(
                    event.data["delta"]["stop_reason"].as_str(),
                );
            }
            _ => {}
        }
    }

    OpenAiChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
        object: "chat.completion".to_string(),
        created: Utc::now().timestamp(),
        model: model.to_string(),
        choices: vec![OpenAiChoice {
            index: 0,
            message: OpenAiAssistantMessage {
                role: "assistant".to_string(),
                content: if text_content.is_empty() {
                    None
                } else {
                    Some(text_content)
                },
                reasoning_content: if reasoning_content.is_empty() {
                    None
                } else {
                    Some(reasoning_content)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            },
            finish_reason,
        }],
        usage: OpenAiUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}

pub fn map_finish_reason(stop_reason: Option<&str>) -> Option<String> {
    Some(
        match stop_reason.unwrap_or("end_turn") {
            "tool_use" => "tool_calls",
            "max_tokens" | "model_context_window_exceeded" => "length",
            _ => "stop",
        }
        .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use crate::kiro::model::events::{
        AssistantResponseEvent, ContextUsageEvent, Event, ToolUseEvent,
    };

    fn assistant_event(content: &str) -> Event {
        Event::AssistantResponse(serde_json::from_value::<AssistantResponseEvent>(
            serde_json::json!({ "content": content }),
        ).unwrap())
    }

    fn tool_use_event(name: &str, tool_use_id: &str, input: &str, stop: bool) -> Event {
        Event::ToolUse(serde_json::from_value::<ToolUseEvent>(
            serde_json::json!({
                "name": name,
                "toolUseId": tool_use_id,
                "input": input,
                "stop": stop
            }),
        ).unwrap())
    }

    fn context_usage_event(percentage: f64) -> Event {
        Event::ContextUsage(serde_json::from_value::<ContextUsageEvent>(
            serde_json::json!({
                "contextUsagePercentage": percentage
            }),
        ).unwrap())
    }

    #[test]
    fn text_only_output_maps_to_openai_chat_completion() {
        let response = super::convert_events_to_response(
            "deepseek-chat",
            12,
            false,
            &[assistant_event("Hello from Kiro")],
        );

        assert_eq!(response.object, "chat.completion");
        assert_eq!(response.model, "deepseek-chat");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("Hello from Kiro")
        );
        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn thinking_output_maps_to_reasoning_content() {
        let response = super::convert_events_to_response(
            "deepseek-reasoner",
            12,
            true,
            &[assistant_event("<thinking>considering</thinking>\n\nFinal answer")],
        );

        assert_eq!(
            response.choices[0].message.reasoning_content.as_deref(),
            Some("considering")
        );
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("Final answer")
        );
    }

    #[test]
    fn tool_use_maps_to_tool_calls_and_finish_reason() {
        let response = super::convert_events_to_response(
            "deepseek-chat",
            12,
            false,
            &[tool_use_event(
                "lookup_weather",
                "call_1",
                "{\"city\":\"Paris\"}",
                true,
            )],
        );

        let tool_calls = response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("tool calls should be present");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_1");
        assert_eq!(tool_calls[0].function.name, "lookup_weather");
        assert_eq!(tool_calls[0].function.arguments, "{\"city\":\"Paris\"}");
        assert_eq!(
            response.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
    }

    #[test]
    fn stop_reason_and_usage_are_mapped() {
        let response = super::convert_events_to_response(
            "deepseek-chat",
            5,
            false,
            &[
                assistant_event("Hello"),
                context_usage_event(25.0),
                Event::Exception {
                    exception_type: "ContentLengthExceededException".to_string(),
                    message: "limit".to_string(),
                },
            ],
        );

        assert_eq!(response.choices[0].finish_reason.as_deref(), Some("length"));
        assert_eq!(response.usage.prompt_tokens, 50_000);
        assert!(response.usage.completion_tokens > 0);
        assert_eq!(
            response.usage.total_tokens,
            response.usage.prompt_tokens + response.usage.completion_tokens
        );
    }
}
