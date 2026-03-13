use std::collections::HashMap;

use chrono::Utc;
use uuid::Uuid;

use crate::{
    anthropic::SseEvent,
    kiro::model::events::{Event, ToolUseEvent},
};

use super::response_converter::map_finish_reason;
use super::types::{
    OpenAiChatCompletionChunk, OpenAiChunkChoice, OpenAiChunkDelta, OpenAiToolCallDelta,
    ToolCallFunction,
};

pub struct OpenAiStreamConverter {
    id: String,
    model: String,
    created: i64,
    tool_call_indices: HashMap<i32, i32>,
    tool_call_indices_by_id: HashMap<String, i32>,
    next_tool_call_index: i32,
    final_finish_reason: Option<String>,
}

impl OpenAiStreamConverter {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
            model: model.into(),
            created: Utc::now().timestamp(),
            tool_call_indices: HashMap::new(),
            tool_call_indices_by_id: HashMap::new(),
            next_tool_call_index: 0,
            final_finish_reason: Some("stop".to_string()),
        }
    }

    pub fn initial_outputs(&self) -> Vec<String> {
        vec![self.serialize_chunk(
            OpenAiChunkDelta {
                role: Some("assistant".to_string()),
                ..Default::default()
            },
            None,
        )]
    }

    pub fn process_kiro_event(&mut self, event: &Event) -> Vec<String> {
        match event {
            Event::AssistantResponse(resp) => self.process_assistant_response(&resp.content),
            Event::ToolUse(tool_use) => self.process_tool_use_event(tool_use),
            Event::Exception { exception_type, .. }
                if exception_type == "ContentLengthExceededException" =>
            {
                self.final_finish_reason = Some("length".to_string());
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    pub fn finish_outputs(&self) -> Vec<String> {
        vec![
            self.serialize_chunk(OpenAiChunkDelta::default(), self.final_finish_reason.clone()),
            "data: [DONE]\n\n".to_string(),
        ]
    }

    pub fn process_sse_event(&mut self, event: &SseEvent) -> Vec<String> {
        match event.event.as_str() {
            "message_start" => vec![self.serialize_chunk(
                OpenAiChunkDelta {
                    role: Some("assistant".to_string()),
                    ..Default::default()
                },
                None,
            )],
            "content_block_start" => {
                if event.data["content_block"]["type"] != "tool_use" {
                    return Vec::new();
                }

                let block_index = event.data["index"].as_i64().unwrap_or_default() as i32;
                let tool_call_index = self.next_tool_call_index;
                self.next_tool_call_index += 1;
                self.tool_call_indices.insert(block_index, tool_call_index);

                vec![self.serialize_chunk(
                    OpenAiChunkDelta {
                        tool_calls: Some(vec![OpenAiToolCallDelta {
                            index: tool_call_index,
                            id: Some(
                                event.data["content_block"]["id"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_string(),
                            ),
                            tool_type: Some("function".to_string()),
                            function: Some(ToolCallFunction {
                                name: event.data["content_block"]["name"]
                                    .as_str()
                                    .unwrap_or_default()
                                    .to_string(),
                                arguments: String::new(),
                            }),
                        }]),
                        ..Default::default()
                    },
                    None,
                )]
            }
            "content_block_delta" => {
                let delta = &event.data["delta"];
                match delta["type"].as_str().unwrap_or_default() {
                    "text_delta" => {
                        let text = delta["text"].as_str().unwrap_or_default();
                        if text.is_empty() {
                            Vec::new()
                        } else {
                            vec![self.serialize_chunk(
                                OpenAiChunkDelta {
                                    content: Some(text.to_string()),
                                    ..Default::default()
                                },
                                None,
                            )]
                        }
                    }
                    "thinking_delta" => {
                        let thinking = delta["thinking"].as_str().unwrap_or_default();
                        if thinking.is_empty() {
                            Vec::new()
                        } else {
                            vec![self.serialize_chunk(
                                OpenAiChunkDelta {
                                    reasoning_content: Some(thinking.to_string()),
                                    ..Default::default()
                                },
                                None,
                            )]
                        }
                    }
                    "input_json_delta" => {
                        let block_index = event.data["index"].as_i64().unwrap_or_default() as i32;
                        let Some(tool_call_index) = self.tool_call_indices.get(&block_index).copied() else {
                            return Vec::new();
                        };

                        vec![self.serialize_chunk(
                            OpenAiChunkDelta {
                                tool_calls: Some(vec![OpenAiToolCallDelta {
                                    index: tool_call_index,
                                    id: None,
                                    tool_type: None,
                                    function: Some(ToolCallFunction {
                                        name: String::new(),
                                        arguments: delta["partial_json"]
                                            .as_str()
                                            .unwrap_or_default()
                                            .to_string(),
                                    }),
                                }]),
                                ..Default::default()
                            },
                            None,
                        )]
                    }
                    _ => Vec::new(),
                }
            }
            "message_delta" => vec![self.serialize_chunk(
                OpenAiChunkDelta::default(),
                map_finish_reason(event.data["delta"]["stop_reason"].as_str()),
            )],
            "message_stop" => vec!["data: [DONE]\n\n".to_string()],
            _ => Vec::new(),
        }
    }

    fn process_assistant_response(&self, content: &str) -> Vec<String> {
        if content.is_empty() {
            return Vec::new();
        }

        vec![self.serialize_chunk(
            OpenAiChunkDelta {
                content: Some(content.to_string()),
                ..Default::default()
            },
            None,
        )]
    }

    fn process_tool_use_event(&mut self, tool_use: &ToolUseEvent) -> Vec<String> {
        let mut delta = OpenAiToolCallDelta {
            index: 0,
            id: None,
            tool_type: None,
            function: None,
        };

        if let Some(index) = self
            .tool_call_indices_by_id
            .get(&tool_use.tool_use_id)
            .copied()
        {
            delta.index = index;
            if !tool_use.input.is_empty() {
                delta.function = Some(ToolCallFunction {
                    name: String::new(),
                    arguments: tool_use.input.clone(),
                });
            }
        } else {
            let index = self.next_tool_call_index;
            self.next_tool_call_index += 1;
            self.tool_call_indices_by_id
                .insert(tool_use.tool_use_id.clone(), index);
            delta.index = index;
            delta.id = Some(tool_use.tool_use_id.clone());
            delta.tool_type = Some("function".to_string());
            delta.function = Some(ToolCallFunction {
                name: tool_use.name.clone(),
                arguments: tool_use.input.clone(),
            });
        }

        if tool_use.stop {
            self.final_finish_reason = Some("tool_calls".to_string());
        }

        vec![self.serialize_chunk(
            OpenAiChunkDelta {
                tool_calls: Some(vec![delta]),
                ..Default::default()
            },
            None,
        )]
    }

    fn serialize_chunk(
        &self,
        delta: OpenAiChunkDelta,
        finish_reason: Option<String>,
    ) -> String {
        let chunk = OpenAiChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![OpenAiChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
        };

        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::anthropic::SseEvent;
    use crate::openai::types::OpenAiChatCompletionChunk;

    fn parse_chunk(payload: &str) -> OpenAiChatCompletionChunk {
        let json = payload
            .strip_prefix("data: ")
            .and_then(|value| value.strip_suffix("\n\n"))
            .expect("chunk must be valid SSE data");
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn emits_first_assistant_role_chunk() {
        let mut converter = super::OpenAiStreamConverter::new("deepseek-chat");
        let outputs = converter.process_sse_event(&SseEvent::new(
            "message_start",
            serde_json::json!({
                "type": "message_start",
                "message": {
                    "usage": { "input_tokens": 1, "output_tokens": 1 }
                }
            }),
        ));

        let chunk = parse_chunk(&outputs[0]);
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices[0].delta.role.as_deref(), Some("assistant"));
    }

    #[test]
    fn emits_text_and_reasoning_deltas() {
        let mut converter = super::OpenAiStreamConverter::new("deepseek-chat");
        let text_outputs = converter.process_sse_event(&SseEvent::new(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": "Hello"
                }
            }),
        ));
        let reasoning_outputs = converter.process_sse_event(&SseEvent::new(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": "Consider this"
                }
            }),
        ));

        assert_eq!(
            parse_chunk(&text_outputs[0]).choices[0].delta.content.as_deref(),
            Some("Hello")
        );
        assert_eq!(
            parse_chunk(&reasoning_outputs[0])
                .choices[0]
                .delta
                .reasoning_content
                .as_deref(),
            Some("Consider this")
        );
    }

    #[test]
    fn emits_tool_call_start_and_argument_deltas() {
        let mut converter = super::OpenAiStreamConverter::new("deepseek-chat");
        let start_outputs = converter.process_sse_event(&SseEvent::new(
            "content_block_start",
            serde_json::json!({
                "type": "content_block_start",
                "index": 3,
                "content_block": {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "lookup_weather",
                    "input": {}
                }
            }),
        ));
        let delta_outputs = converter.process_sse_event(&SseEvent::new(
            "content_block_delta",
            serde_json::json!({
                "type": "content_block_delta",
                "index": 3,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "{\"city\":\"Paris\"}"
                }
            }),
        ));

        let start_chunk = parse_chunk(&start_outputs[0]);
        let start_tool_call = start_chunk.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(start_tool_call[0].id.as_deref(), Some("call_1"));
        assert_eq!(
            start_tool_call[0]
                .function
                .as_ref()
                .map(|function| function.name.as_str()),
            Some("lookup_weather")
        );

        let delta_chunk = parse_chunk(&delta_outputs[0]);
        let delta_tool_call = delta_chunk.choices[0].delta.tool_calls.as_ref().unwrap();
        assert_eq!(
            delta_tool_call[0]
                .function
                .as_ref()
                .map(|function| function.arguments.as_str()),
            Some("{\"city\":\"Paris\"}")
        );
    }

    #[test]
    fn emits_finish_chunk_and_done_marker() {
        let mut converter = super::OpenAiStreamConverter::new("deepseek-chat");
        let finish_outputs = converter.process_sse_event(&SseEvent::new(
            "message_delta",
            serde_json::json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use"
                },
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5
                }
            }),
        ));
        let done_outputs = converter.process_sse_event(&SseEvent::new(
            "message_stop",
            serde_json::json!({ "type": "message_stop" }),
        ));

        let finish_chunk = parse_chunk(&finish_outputs[0]);
        assert_eq!(
            finish_chunk.choices[0].finish_reason.as_deref(),
            Some("tool_calls")
        );
        assert_eq!(done_outputs[0], "data: [DONE]\n\n");
    }
}
