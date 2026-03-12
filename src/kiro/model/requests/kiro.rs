//! Kiro 请求类型定义
//!
//! 定义 Kiro API 的主请求结构

use serde::{Deserialize, Serialize};

use super::conversation::ConversationState;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

impl InferenceConfig {
    pub fn is_empty(&self) -> bool {
        self.max_tokens.is_none() && self.temperature.is_none() && self.top_p.is_none()
    }
}

fn inference_config_is_empty(config: &Option<InferenceConfig>) -> bool {
    match config {
        Some(config) => config.is_empty(),
        None => true,
    }
}

/// Kiro API 请求
///
/// 用于构建发送给 Kiro API 的请求
///
/// # 示例
///
/// ```rust
/// use kiro_rs::kiro::model::requests::{
///     KiroRequest, ConversationState, CurrentMessage, UserInputMessage, Tool
/// };
///
/// // 创建简单请求
/// let state = ConversationState::new("conv-123")
///     .with_agent_task_type("vibe")
///     .with_current_message(CurrentMessage::new(
///         UserInputMessage::new("Hello", "claude-3-5-sonnet")
///     ));
///
/// let request = KiroRequest::new(state);
/// let json = request.to_json().unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KiroRequest {
    /// 对话状态
    pub conversation_state: ConversationState,
    /// Profile ARN（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile_arn: Option<String>,
    /// 推理配置（可选）
    #[serde(default, skip_serializing_if = "inference_config_is_empty")]
    pub inference_config: Option<InferenceConfig>,
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kiro::model::requests::conversation::{
        ConversationState, CurrentMessage, UserInputMessage,
    };

    fn sample_request() -> KiroRequest {
        KiroRequest {
            conversation_state: ConversationState::new("conv-456")
                .with_current_message(CurrentMessage::new(UserInputMessage::new(
                    "Test message",
                    "claude-sonnet-4.6",
                ))),
            profile_arn: None,
            inference_config: None,
        }
    }

    #[test]
    fn test_kiro_request_deserialize() {
        let json = r#"{
            "conversationState": {
                "conversationId": "conv-456",
                "currentMessage": {
                    "userInputMessage": {
                        "content": "Test message",
                        "modelId": "claude-3-5-sonnet",
                        "userInputMessageContext": {}
                    }
                }
            }
        }"#;

        let request: KiroRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.conversation_state.conversation_id, "conv-456");
        assert_eq!(
            request
                .conversation_state
                .current_message
                .user_input_message
                .content,
            "Test message"
        );
    }

    #[test]
    fn kiro_request_inference_config_serializes_when_present() {
        let mut request = sample_request();
        request.inference_config = Some(InferenceConfig {
            max_tokens: Some(2048),
            temperature: Some(0.2),
            top_p: Some(0.9),
        });

        let json = serde_json::to_value(&request).unwrap();

        assert_eq!(json["inferenceConfig"]["maxTokens"], 2048);
        assert_eq!(json["inferenceConfig"]["temperature"], 0.2);
        assert_eq!(json["inferenceConfig"]["topP"], 0.9);
    }

    #[test]
    fn kiro_request_inference_config_omitted_when_empty() {
        let mut request = sample_request();
        request.inference_config = Some(InferenceConfig::default());

        let json = serde_json::to_value(&request).unwrap();

        assert!(json.get("inferenceConfig").is_none());
    }
}
