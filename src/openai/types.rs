use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionsRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionTool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    pub usage: OpenAiUsage,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiChoice {
    pub index: i32,
    pub message: OpenAiAssistantMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiAssistantMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiUsage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<OpenAiChunkChoice>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiChunkChoice {
    pub index: i32,
    pub delta: OpenAiChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OpenAiChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCallDelta>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiToolCallDelta {
    pub index: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunction>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiErrorResponse {
    pub error: OpenAiError,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
}
