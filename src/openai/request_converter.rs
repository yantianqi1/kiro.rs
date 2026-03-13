use std::collections::HashSet;

use uuid::Uuid;

use crate::kiro::model::requests::conversation::{
    AssistantMessage, ConversationState, CurrentMessage, HistoryAssistantMessage,
    HistoryUserMessage, Message, UserInputMessage, UserInputMessageContext, UserMessage,
};
use crate::kiro::model::requests::kiro::{InferenceConfig, KiroRequest};
use crate::kiro::model::requests::tool::{
    InputSchema, Tool, ToolResult, ToolSpecification, ToolUseEntry,
};

use super::types::{ChatCompletionsRequest, ChatMessage, ToolDefinition};

const DEFAULT_THINKING_BUDGET: i32 = 24_576;
const KIRO_MODEL_DEEPSEEK_3_2: &str = "deepseek-3.2";

#[derive(Debug)]
pub enum ConversionError {
    UnsupportedModel(String),
    EmptyMessages,
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedModel(model) => write!(f, "unsupported model: {model}"),
            Self::EmptyMessages => write!(f, "messages cannot be empty"),
        }
    }
}

impl std::error::Error for ConversionError {}

fn is_unified_deepseek_model(model: &str) -> bool {
    matches!(
        model,
        "deepseek-v3.2-exp"
            | "deepseek-v3-2-exp"
            | "deepseek-chat"
            | "deepseek-reasoner"
            | "deepseek-3-2"
            | "deepseek-3-2-thinking"
            | "deepseek-3.2"
            | "deepseek-3.2-thinking"
            | "deepseek"
    ) || model.contains("deepseek-r1")
}

pub fn convert_request(req: &ChatCompletionsRequest) -> Result<KiroRequest, ConversionError> {
    if req.messages.is_empty() {
        return Err(ConversionError::EmptyMessages);
    }

    let (model_id, model_implies_thinking) = map_model(&req.model)
        .ok_or_else(|| ConversionError::UnsupportedModel(req.model.clone()))?;
    let last_user_index = req
        .messages
        .iter()
        .rposition(|message| message.role == "user")
        .ok_or(ConversionError::EmptyMessages)?;

    let thinking_prefix = build_thinking_prefix(req, model_implies_thinking);
    let mut history = build_system_history(&req.messages, &model_id, thinking_prefix);
    let mut known_tool_use_ids = HashSet::new();
    let mut paired_tool_use_ids = HashSet::new();
    let mut pending_tool_results = Vec::new();

    for message in req.messages.iter().take(last_user_index) {
        match message.role.as_str() {
            "system" => {}
            "user" => {
                let user_message =
                    build_history_user_message(message, &model_id, std::mem::take(&mut pending_tool_results));
                history.push(Message::User(user_message));
            }
            "assistant" => {
                let (assistant_message, tool_use_ids) = build_history_assistant_message(message);
                known_tool_use_ids.extend(tool_use_ids);
                history.push(Message::Assistant(assistant_message));
            }
            "tool" => {
                if let Some(tool_result) =
                    build_tool_result(message, &known_tool_use_ids, &paired_tool_use_ids)
                {
                    paired_tool_use_ids.insert(tool_result.tool_use_id.clone());
                    pending_tool_results.push(tool_result);
                }
            }
            _ => {}
        }
    }

    let tools = build_tools(req.tools.as_ref(), &history);
    let current_message = req
        .messages
        .get(last_user_index)
        .expect("last_user_index should always resolve to an existing message");
    let current_content = extract_message_text(current_message);
    let current_context = UserInputMessageContext::new()
        .with_tools(tools)
        .with_tool_results(pending_tool_results);
    let current_user_input = UserInputMessage::new(current_content, &model_id)
        .with_context(current_context)
        .with_origin("AI_EDITOR");

    remove_orphaned_tool_uses(&mut history, &known_tool_use_ids, &paired_tool_use_ids);

    Ok(KiroRequest {
        conversation_state: ConversationState::new(Uuid::new_v4().to_string())
            .with_agent_continuation_id(Uuid::new_v4().to_string())
            .with_agent_task_type("vibe")
            .with_chat_trigger_type("MANUAL")
            .with_current_message(CurrentMessage::new(current_user_input))
            .with_history(history),
        profile_arn: None,
        inference_config: build_inference_config(req),
    })
}

fn map_model(model: &str) -> Option<(String, bool)> {
    let model = model.to_lowercase();

    if is_unified_deepseek_model(&model) || model.contains("deepseek-v3") {
        return Some((KIRO_MODEL_DEEPSEEK_3_2.to_string(), true));
    }
    if model.contains("sonnet") {
        if model.contains("4-6") || model.contains("4.6") {
            return Some(("claude-sonnet-4.6".to_string(), model.contains("thinking")));
        }
        return Some(("claude-sonnet-4.5".to_string(), model.contains("thinking")));
    }
    if model.contains("opus") {
        if model.contains("4-5") || model.contains("4.5") {
            return Some(("claude-opus-4.5".to_string(), model.contains("thinking")));
        }
        return Some(("claude-opus-4.6".to_string(), model.contains("thinking")));
    }
    if model.contains("haiku") {
        return Some(("claude-haiku-4.5".to_string(), model.contains("thinking")));
    }

    None
}

fn build_inference_config(req: &ChatCompletionsRequest) -> Option<InferenceConfig> {
    let config = InferenceConfig {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
    };

    if config.is_empty() {
        None
    } else {
        Some(config)
    }
}

fn build_system_history(
    messages: &[ChatMessage],
    model_id: &str,
    thinking_prefix: Option<String>,
) -> Vec<Message> {
    let system_content = messages
        .iter()
        .filter(|message| message.role == "system")
        .map(extract_message_text)
        .filter(|content| !content.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    let final_content = if !system_content.is_empty() {
        match thinking_prefix {
            Some(prefix) if !contains_thinking_tags(&system_content) => {
                format!("{prefix}\n{system_content}")
            }
            _ => system_content,
        }
    } else if let Some(prefix) = thinking_prefix {
        prefix
    } else {
        String::new()
    };

    if final_content.is_empty() {
        return Vec::new();
    }

    vec![
        Message::User(HistoryUserMessage::new(final_content, model_id)),
        Message::Assistant(HistoryAssistantMessage::new(
            "I will follow these instructions.",
        )),
    ]
}

fn build_thinking_prefix(
    req: &ChatCompletionsRequest,
    model_implies_thinking: bool,
) -> Option<String> {
    if request_already_contains_thinking_tags(req) {
        return None;
    }

    if let Some(effort) = req
        .reasoning_effort
        .as_deref()
        .filter(|effort| !effort.eq_ignore_ascii_case("none"))
    {
        return Some(format!(
            "<thinking_mode>adaptive</thinking_mode><thinking_effort>{effort}</thinking_effort>"
        ));
    }

    if model_implies_thinking {
        return Some(format!(
            "<thinking_mode>enabled</thinking_mode><max_thinking_length>{DEFAULT_THINKING_BUDGET}</max_thinking_length>"
        ));
    }

    None
}

fn request_already_contains_thinking_tags(req: &ChatCompletionsRequest) -> bool {
    req.messages.iter().any(|message| {
        let content = extract_message_text(message);
        contains_thinking_tags(&content)
    })
}

fn contains_thinking_tags(content: &str) -> bool {
    content.contains("<thinking_mode>")
        || content.contains("<thinking_effort>")
        || content.contains("<max_thinking_length>")
}

fn build_history_user_message(
    message: &ChatMessage,
    model_id: &str,
    tool_results: Vec<ToolResult>,
) -> HistoryUserMessage {
    let mut user_message = UserMessage::new(extract_message_text(message), model_id);

    if !tool_results.is_empty() {
        user_message = user_message.with_context(
            UserInputMessageContext::new().with_tool_results(tool_results),
        );
    }

    HistoryUserMessage { user_input_message: user_message }
}

fn build_history_assistant_message(message: &ChatMessage) -> (HistoryAssistantMessage, HashSet<String>) {
    let tool_uses = message
        .tool_calls
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter(|tool_call| tool_call.call_type == "function")
        .map(|tool_call| {
            let input = serde_json::from_str(&tool_call.function.arguments)
                .unwrap_or_else(|_| serde_json::json!({}));
            ToolUseEntry::new(tool_call.id, tool_call.function.name).with_input(input)
        })
        .collect::<Vec<_>>();

    let content = {
        let content = extract_message_text(message);
        if content.is_empty() && !tool_uses.is_empty() {
            " ".to_string()
        } else if content.is_empty() {
            " ".to_string()
        } else {
            content
        }
    };

    let tool_use_ids = tool_uses
        .iter()
        .map(|tool_use| tool_use.tool_use_id.clone())
        .collect::<HashSet<_>>();
    let mut assistant_message = AssistantMessage::new(content);
    if !tool_uses.is_empty() {
        assistant_message = assistant_message.with_tool_uses(tool_uses);
    }

    (
        HistoryAssistantMessage {
            assistant_response_message: assistant_message,
        },
        tool_use_ids,
    )
}

fn build_tool_result(
    message: &ChatMessage,
    known_tool_use_ids: &HashSet<String>,
    paired_tool_use_ids: &HashSet<String>,
) -> Option<ToolResult> {
    let tool_use_id = message.tool_call_id.as_ref()?;
    if !known_tool_use_ids.contains(tool_use_id) || paired_tool_use_ids.contains(tool_use_id) {
        return None;
    }

    Some(ToolResult::success(
        tool_use_id.clone(),
        extract_message_text(message),
    ))
}

fn build_tools(tools: Option<&Vec<ToolDefinition>>, history: &[Message]) -> Vec<Tool> {
    let mut converted = tools
        .into_iter()
        .flat_map(|tools| tools.iter())
        .filter(|tool| tool.tool_type == "function")
        .map(|tool| Tool {
            tool_specification: ToolSpecification {
                name: tool.function.name.clone(),
                description: tool.function.description.clone().unwrap_or_default(),
                input_schema: InputSchema::from_json(normalize_json_schema(
                    tool.function.parameters.clone(),
                )),
            },
        })
        .collect::<Vec<_>>();

    let existing_names = converted
        .iter()
        .map(|tool| tool.tool_specification.name.to_lowercase())
        .collect::<HashSet<_>>();

    for tool_name in collect_history_tool_names(history) {
        if !existing_names.contains(&tool_name.to_lowercase()) {
            converted.push(placeholder_tool(&tool_name));
        }
    }

    converted
}

fn collect_history_tool_names(history: &[Message]) -> Vec<String> {
    let mut tool_names = Vec::new();

    for message in history {
        if let Message::Assistant(assistant_message) = message {
            if let Some(tool_uses) = &assistant_message.assistant_response_message.tool_uses {
                for tool_use in tool_uses {
                    if !tool_names.contains(&tool_use.name) {
                        tool_names.push(tool_use.name.clone());
                    }
                }
            }
        }
    }

    tool_names
}

fn placeholder_tool(name: &str) -> Tool {
    Tool {
        tool_specification: ToolSpecification {
            name: name.to_string(),
            description: "Tool used in conversation history".to_string(),
            input_schema: InputSchema::from_json(serde_json::json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": true
            })),
        },
    }
}

fn remove_orphaned_tool_uses(
    history: &mut [Message],
    known_tool_use_ids: &HashSet<String>,
    paired_tool_use_ids: &HashSet<String>,
) {
    let orphaned_tool_use_ids = known_tool_use_ids
        .difference(paired_tool_use_ids)
        .cloned()
        .collect::<HashSet<_>>();

    if orphaned_tool_use_ids.is_empty() {
        return;
    }

    for message in history.iter_mut() {
        if let Message::Assistant(assistant_message) = message {
            if let Some(tool_uses) = assistant_message
                .assistant_response_message
                .tool_uses
                .as_mut()
            {
                tool_uses.retain(|tool_use| !orphaned_tool_use_ids.contains(&tool_use.tool_use_id));
                if tool_uses.is_empty() {
                    assistant_message.assistant_response_message.tool_uses = None;
                }
            }
        }
    }
}

fn normalize_json_schema(schema: serde_json::Value) -> serde_json::Value {
    let serde_json::Value::Object(mut object) = schema else {
        return serde_json::json!({
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": true
        });
    };

    if !object
        .get("type")
        .and_then(|value| value.as_str())
        .is_some_and(|value| !value.is_empty())
    {
        object.insert("type".to_string(), serde_json::Value::String("object".to_string()));
    }

    match object.get("properties") {
        Some(serde_json::Value::Object(_)) => {}
        _ => {
            object.insert(
                "properties".to_string(),
                serde_json::Value::Object(serde_json::Map::new()),
            );
        }
    }

    let required = match object.remove("required") {
        Some(serde_json::Value::Array(values)) => serde_json::Value::Array(
            values
                .into_iter()
                .filter_map(|value| value.as_str().map(|value| value.into()))
                .collect(),
        ),
        _ => serde_json::Value::Array(Vec::new()),
    };
    object.insert("required".to_string(), required);

    match object.get("additionalProperties") {
        Some(serde_json::Value::Bool(_)) | Some(serde_json::Value::Object(_)) => {}
        _ => {
            object.insert(
                "additionalProperties".to_string(),
                serde_json::Value::Bool(true),
            );
        }
    }

    serde_json::Value::Object(object)
}

fn extract_message_text(message: &ChatMessage) -> String {
    match &message.content {
        Some(serde_json::Value::String(content)) => content.clone(),
        Some(serde_json::Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                part.get("text")
                    .and_then(|text| text.as_str())
                    .map(|text| text.to_string())
                    .or_else(|| {
                        part.get("content")
                            .and_then(|text| text.as_str())
                            .map(|text| text.to_string())
                    })
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(serde_json::Value::Null) | None => String::new(),
        Some(content) => content.as_str().map(ToString::to_string).unwrap_or_else(|| content.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use crate::kiro::model::requests::conversation::Message as KiroMessage;

    use crate::openai::types::{
        ChatCompletionsRequest, ChatMessage, FunctionTool, ToolCall, ToolCallFunction,
        ToolDefinition,
    };

    fn user_message(content: &str) -> ChatMessage {
        ChatMessage {
            role: "user".to_string(),
            content: Some(serde_json::json!(content)),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    #[test]
    fn deepseek_plain_model_maps_to_kiro_and_inference_config() {
        let request = ChatCompletionsRequest {
            model: "deepseek-v3.2-exp".to_string(),
            messages: vec![user_message("Hello from OpenAI")],
            stream: false,
            max_tokens: Some(2048),
            temperature: Some(0.2),
            top_p: Some(0.95),
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: None,
        };

        let kiro_request = super::convert_request(&request).unwrap();

        assert_eq!(
            kiro_request
                .conversation_state
                .current_message
                .user_input_message
                .model_id,
            "deepseek-3.2"
        );
        assert_eq!(
            kiro_request
                .conversation_state
                .current_message
                .user_input_message
                .origin
                .as_deref(),
            Some("AI_EDITOR")
        );

        let inference = kiro_request
            .inference_config
            .expect("inference config should be present");
        assert_eq!(inference.max_tokens, Some(2048));
        assert_eq!(inference.temperature, Some(0.2));
        assert_eq!(inference.top_p, Some(0.95));
    }

    #[test]
    fn deepseek_public_model_defaults_to_thinking() {
        let request = ChatCompletionsRequest {
            model: "deepseek-v3.2-exp".to_string(),
            messages: vec![user_message("Think carefully about this")],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: None,
        };

        let kiro_request = super::convert_request(&request).unwrap();
        let KiroMessage::User(system_message) = &kiro_request.conversation_state.history[0] else {
            panic!("expected synthetic system message");
        };

        let content = &system_message.user_input_message.content;
        assert!(content.contains("<thinking_mode>enabled</thinking_mode>"));
        assert!(content.contains("<max_thinking_length>24576</max_thinking_length>"));
    }

    #[test]
    fn deepseek_legacy_aliases_map_to_same_model_and_default_thinking() {
        for alias in [
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-3-2",
            "deepseek-3-2-thinking",
        ] {
            assert_eq!(
                super::map_model(alias),
                Some(("deepseek-3.2".to_string(), true)),
                "alias {alias} should map to the unified deepseek family"
            );
        }
    }

    #[test]
    fn deepseek_reasoning_effort_injects_thinking_tags_once() {
        let request = ChatCompletionsRequest {
            model: "deepseek-reasoner".to_string(),
            messages: vec![user_message("Think carefully about this")],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: Some("high".to_string()),
        };

        let kiro_request = super::convert_request(&request).unwrap();
        let KiroMessage::User(system_message) = &kiro_request.conversation_state.history[0] else {
            panic!("expected synthetic system message");
        };

        let content = &system_message.user_input_message.content;
        assert!(content.contains("<thinking_mode>adaptive</thinking_mode>"));
        assert!(content.contains("<thinking_effort>high</thinking_effort>"));
        assert_eq!(content.matches("<thinking_mode>").count(), 1);
    }

    #[test]
    fn assistant_tool_calls_and_tool_results_attach_to_current_user() {
        let request = ChatCompletionsRequest {
            model: "deepseek-v3.2-exp".to_string(),
            messages: vec![
                user_message("Check the weather"),
                ChatMessage {
                    role: "assistant".to_string(),
                    content: None,
                    name: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".to_string(),
                        call_type: "function".to_string(),
                        function: ToolCallFunction {
                            name: "lookup_weather".to_string(),
                            arguments: "{\"city\":\"Hangzhou\"}".to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                ChatMessage {
                    role: "tool".to_string(),
                    content: Some(serde_json::json!("Sunny")),
                    name: Some("lookup_weather".to_string()),
                    tool_calls: None,
                    tool_call_id: Some("call_1".to_string()),
                },
                user_message("Continue"),
            ],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: Some(vec![ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionTool {
                    name: "lookup_weather".to_string(),
                    description: Some("Look up weather".to_string()),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" }
                        },
                        "required": ["city"]
                    }),
                },
            }]),
            tool_choice: None,
            response_format: None,
            reasoning_effort: None,
        };

        let kiro_request = super::convert_request(&request).unwrap();
        let history = &kiro_request.conversation_state.history;

        let assistant_message = history
            .iter()
            .find_map(|message| match message {
                KiroMessage::Assistant(assistant_message)
                    if assistant_message.assistant_response_message.tool_uses.is_some() =>
                {
                    Some(assistant_message)
                }
                _ => None,
            })
            .expect("expected assistant history entry with tool uses");
        let tool_uses = assistant_message
            .assistant_response_message
            .tool_uses
            .as_ref()
            .expect("tool uses should be present");
        assert_eq!(tool_uses.len(), 1);
        assert_eq!(tool_uses[0].tool_use_id, "call_1");
        assert_eq!(tool_uses[0].name, "lookup_weather");
        assert_eq!(tool_uses[0].input["city"], "Hangzhou");
        assert_eq!(assistant_message.assistant_response_message.content, " ");

        let current = &kiro_request.conversation_state.current_message.user_input_message;
        assert_eq!(current.user_input_message_context.tool_results.len(), 1);
        assert_eq!(
            current.user_input_message_context.tool_results[0].tool_use_id,
            "call_1"
        );
        assert_eq!(
            current.user_input_message_context.tool_results[0].content[0]["text"],
            "Sunny"
        );
        assert_eq!(current.user_input_message_context.tools.len(), 1);
        assert_eq!(
            current.user_input_message_context.tools[0]
                .tool_specification
                .name,
            "lookup_weather"
        );
    }

    #[test]
    fn orphaned_tool_results_are_dropped() {
        let request = ChatCompletionsRequest {
            model: "deepseek-v3.2-exp".to_string(),
            messages: vec![
                user_message("Hello"),
                ChatMessage {
                    role: "tool".to_string(),
                    content: Some(serde_json::json!("orphaned")),
                    name: Some("lookup_weather".to_string()),
                    tool_calls: None,
                    tool_call_id: Some("missing_call".to_string()),
                },
                user_message("Continue"),
            ],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            reasoning_effort: None,
        };

        let kiro_request = super::convert_request(&request).unwrap();
        assert!(
            kiro_request
                .conversation_state
                .current_message
                .user_input_message
                .user_input_message_context
                .tool_results
                .is_empty()
        );
    }
}
