use std::{convert::Infallible, future::Future, pin::Pin};

use anyhow::Error;
use axum::{
    Json as JsonExtractor,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
};
use bytes::Bytes;
use futures::{Stream, StreamExt, stream};

use crate::{
    anthropic::{AppState, SseEvent, StreamContext},
    kiro::{model::events::Event, parser::decoder::EventStreamDecoder},
    token,
};

use super::{
    request_converter::{self, ConversionError},
    response_converter::convert_events_to_response,
    stream::OpenAiStreamConverter,
    types::{ChatCompletionsRequest, OpenAiError, OpenAiErrorResponse},
};

type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, Error>> + Send>>;
type OutputStream = Pin<Box<dyn Stream<Item = Result<Bytes, Infallible>> + Send>>;

struct PreparedRequest {
    body: String,
    model: String,
    input_tokens: i32,
    thinking_enabled: bool,
}

pub async fn post_chat_completions(
    State(state): State<AppState>,
    JsonExtractor(payload): JsonExtractor<ChatCompletionsRequest>,
) -> Response {
    let provider = match &state.kiro_provider {
        Some(provider) => provider.clone(),
        None => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "api_error",
                "Kiro API provider not configured",
            );
        }
    };

    if payload.stream {
        handle_stream_with_transport(payload, state.profile_arn.clone(), move |request_body| {
            let provider = provider.clone();
            async move {
                let response = provider.call_api_stream(&request_body).await?;
                let stream = response.bytes_stream().map(|chunk| chunk.map_err(Error::from));
                Ok::<ByteStream, Error>(Box::pin(stream))
            }
        })
        .await
    } else {
        handle_non_stream_with_transport(
            payload,
            state.profile_arn.clone(),
            move |request_body| {
                let provider = provider.clone();
                async move {
                    let response = provider.call_api(&request_body).await?;
                    let bytes = response.bytes().await?;
                    Ok::<Bytes, Error>(bytes)
                }
            },
        )
        .await
    }
}

async fn handle_non_stream_with_transport<F, Fut>(
    payload: ChatCompletionsRequest,
    profile_arn: Option<String>,
    fetch: F,
) -> Response
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = Result<Bytes, Error>>,
{
    let prepared = match prepare_request(payload, profile_arn) {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };

    let body_bytes = match fetch(prepared.body).await {
        Ok(body) => body,
        Err(error) => return map_provider_error(error),
    };

    let events = decode_event_stream_bytes(&body_bytes);
    let response = convert_events_to_response(
        &prepared.model,
        prepared.input_tokens,
        prepared.thinking_enabled,
        &events,
    );

    (StatusCode::OK, Json(response)).into_response()
}

async fn handle_stream_with_transport<F, Fut>(
    payload: ChatCompletionsRequest,
    profile_arn: Option<String>,
    fetch: F,
) -> Response
where
    F: FnOnce(String) -> Fut,
    Fut: Future<Output = Result<ByteStream, Error>>,
{
    let prepared = match prepare_request(payload, profile_arn) {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };

    let upstream = match fetch(prepared.body).await {
        Ok(stream) => stream,
        Err(error) => return map_provider_error(error),
    };

    let stream = create_openai_sse_stream(
        upstream,
        prepared.model,
        prepared.input_tokens,
        prepared.thinking_enabled,
    );

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}

fn prepare_request(
    payload: ChatCompletionsRequest,
    profile_arn: Option<String>,
) -> Result<PreparedRequest, Response> {
    let model = payload.model.clone();
    let thinking_enabled = is_reasoning_enabled(&payload);
    let input_tokens = estimate_input_tokens(&payload);

    let mut kiro_request = match request_converter::convert_request(&payload) {
        Ok(request) => request,
        Err(ConversionError::UnsupportedModel(model)) => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Unsupported model: {model}"),
            ));
        }
        Err(ConversionError::EmptyMessages) => {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "messages cannot be empty",
            ));
        }
    };

    kiro_request.profile_arn = profile_arn;

    let body = match serde_json::to_string(&kiro_request) {
        Ok(body) => body,
        Err(error) => {
            return Err(error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                format!("Failed to serialize request: {error}"),
            ));
        }
    };

    Ok(PreparedRequest {
        body,
        model,
        input_tokens,
        thinking_enabled,
    })
}

fn estimate_input_tokens(payload: &ChatCompletionsRequest) -> i32 {
    let mut total = 0;

    for message in &payload.messages {
        total += token::count_tokens(&message.role) as i32;
        if let Some(name) = &message.name {
            total += token::count_tokens(name) as i32;
        }
        if let Some(content) = &message.content {
            total += token::count_tokens(&content.to_string()) as i32;
        }
        if let Some(tool_calls) = &message.tool_calls {
            for tool_call in tool_calls {
                total += token::count_tokens(&tool_call.function.name) as i32;
                total += token::count_tokens(&tool_call.function.arguments) as i32;
            }
        }
    }

    if let Some(tools) = &payload.tools {
        for tool in tools {
            total += token::count_tokens(&tool.function.name) as i32;
            if let Some(description) = &tool.function.description {
                total += token::count_tokens(description) as i32;
            }
            total += token::count_tokens(&tool.function.parameters.to_string()) as i32;
        }
    }

    total.max(1)
}

fn is_reasoning_enabled(payload: &ChatCompletionsRequest) -> bool {
    payload
        .reasoning_effort
        .as_deref()
        .is_some_and(|effort| !effort.eq_ignore_ascii_case("none"))
        || payload.model.to_lowercase().contains("reasoner")
        || payload.model.to_lowercase().contains("thinking")
        || payload.messages.iter().any(|message| {
            let content = message
                .content
                .as_ref()
                .map(|content| content.to_string())
                .unwrap_or_default();
            content.contains("<thinking>") || content.contains("<thinking_mode>")
        })
}

fn decode_event_stream_bytes(body_bytes: &[u8]) -> Vec<Event> {
    let mut decoder = EventStreamDecoder::new();
    if let Err(error) = decoder.feed(body_bytes) {
        tracing::warn!("failed to feed event stream decoder: {error}");
    }

    decoder
        .decode_iter()
        .filter_map(|result| match result {
            Ok(frame) => Event::from_frame(frame).ok(),
            Err(error) => {
                tracing::warn!("failed to decode event stream frame: {error}");
                None
            }
        })
        .collect()
}

fn create_openai_sse_stream(
    upstream: ByteStream,
    model: String,
    input_tokens: i32,
    thinking_enabled: bool,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    let stream_context = StreamContext::new_with_thinking(model.clone(), input_tokens, thinking_enabled);
    let openai_converter = OpenAiStreamConverter::new(model);

    stream::unfold(
        (
            upstream,
            EventStreamDecoder::new(),
            stream_context,
            openai_converter,
            false,
            false,
        ),
        |(mut upstream, mut decoder, mut stream_context, mut converter, initial_sent, finished)| async move {
            if finished {
                return None;
            }

            if !initial_sent {
                let initial_outputs = convert_sse_events(
                    &mut converter,
                    stream_context.generate_initial_events(),
                );
                return Some((
                    output_stream_from_chunks(initial_outputs),
                    (upstream, decoder, stream_context, converter, true, false),
                ));
            }

            match upstream.next().await {
                Some(Ok(chunk)) => {
                    if let Err(error) = decoder.feed(&chunk) {
                        tracing::warn!("failed to feed stream decoder: {error}");
                    }

                    let mut outputs = Vec::new();
                    for frame in decoder.decode_iter().flatten() {
                        if let Ok(event) = Event::from_frame(frame) {
                            outputs.extend(convert_sse_events(
                                &mut converter,
                                stream_context.process_kiro_event(&event),
                            ));
                        }
                    }

                    Some((
                        output_stream_from_chunks(outputs),
                        (upstream, decoder, stream_context, converter, true, false),
                    ))
                }
                Some(Err(error)) => {
                    tracing::warn!("upstream stream error: {error}");
                    let final_outputs = convert_sse_events(
                        &mut converter,
                        stream_context.generate_final_events(),
                    );
                    Some((
                        output_stream_from_chunks(final_outputs),
                        (upstream, decoder, stream_context, converter, true, true),
                    ))
                }
                None => {
                    let final_outputs = convert_sse_events(
                        &mut converter,
                        stream_context.generate_final_events(),
                    );
                    Some((
                        output_stream_from_chunks(final_outputs),
                        (upstream, decoder, stream_context, converter, true, true),
                    ))
                }
            }
        },
    )
    .flatten()
}

fn convert_sse_events(
    converter: &mut OpenAiStreamConverter,
    sse_events: Vec<SseEvent>,
) -> Vec<String> {
    let mut outputs = Vec::new();
    for event in sse_events {
        outputs.extend(converter.process_sse_event(&event));
    }
    outputs
}

fn output_stream_from_chunks(chunks: Vec<String>) -> OutputStream {
    Box::pin(stream::iter(
        chunks
            .into_iter()
            .map(|chunk| Ok(Bytes::from(chunk)))
            .collect::<Vec<_>>(),
    ))
}

fn map_provider_error(error: Error) -> Response {
    let error_message = error.to_string();

    if error_message.contains("CONTENT_LENGTH_EXCEEDS_THRESHOLD")
        || error_message.contains("Input is too long")
    {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            error_message,
        );
    }

    error_response(StatusCode::BAD_GATEWAY, "api_error", error_message)
}

fn error_response(
    status: StatusCode,
    error_type: impl Into<String>,
    message: impl Into<String>,
) -> Response {
    (
        status,
        Json(OpenAiErrorResponse {
            error: OpenAiError {
                message: message.into(),
                error_type: error_type.into(),
            },
        }),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use anyhow::Error;
    use axum::{
        body::to_bytes,
        http::{StatusCode, header},
        response::Response,
    };
    use bytes::Bytes;
    use futures::stream;

    use crate::{
        anthropic::AppState,
        kiro::parser::crc::crc32,
        openai::{
            router::create_router,
            types::{ChatCompletionsRequest, ChatMessage},
        },
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

    fn encode_string_header(name: &str, value: &str, buffer: &mut Vec<u8>) {
        buffer.push(name.len() as u8);
        buffer.extend_from_slice(name.as_bytes());
        buffer.push(7);
        buffer.extend_from_slice(&(value.len() as u16).to_be_bytes());
        buffer.extend_from_slice(value.as_bytes());
    }

    fn encode_event_frame(event_type: &str, payload: serde_json::Value) -> Vec<u8> {
        let mut headers = Vec::new();
        encode_string_header(":message-type", "event", &mut headers);
        encode_string_header(":event-type", event_type, &mut headers);

        let payload = serde_json::to_vec(&payload).unwrap();
        let total_length = 12 + headers.len() + payload.len() + 4;
        let mut frame = Vec::with_capacity(total_length);
        frame.extend_from_slice(&(total_length as u32).to_be_bytes());
        frame.extend_from_slice(&(headers.len() as u32).to_be_bytes());
        let prelude_crc = crc32(&frame);
        frame.extend_from_slice(&prelude_crc.to_be_bytes());
        frame.extend_from_slice(&headers);
        frame.extend_from_slice(&payload);
        let message_crc = crc32(&frame);
        frame.extend_from_slice(&message_crc.to_be_bytes());
        frame
    }

    fn sample_non_stream_bytes() -> Bytes {
        let mut frames = Vec::new();
        frames.extend_from_slice(&encode_event_frame(
            "assistantResponseEvent",
            serde_json::json!({ "content": "Hello from upstream" }),
        ));
        frames.extend_from_slice(&encode_event_frame(
            "contextUsageEvent",
            serde_json::json!({ "contextUsagePercentage": 5.0 }),
        ));
        Bytes::from(frames)
    }

    fn sample_stream_bytes() -> Bytes {
        let mut frames = Vec::new();
        frames.extend_from_slice(&encode_event_frame(
            "assistantResponseEvent",
            serde_json::json!({ "content": "Hello" }),
        ));
        Bytes::from(frames)
    }

    #[tokio::test]
    async fn non_stream_request_returns_openai_json_and_calls_transport() {
        let captured_body = Arc::new(Mutex::new(None::<String>));
        let captured_body_ref = captured_body.clone();
        let response: Response = super::handle_non_stream_with_transport(
            ChatCompletionsRequest {
                model: "deepseek-chat".to_string(),
                messages: vec![user_message("Hello")],
                stream: false,
                max_tokens: Some(32),
                temperature: None,
                top_p: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                reasoning_effort: None,
            },
            Some("arn:aws:test".to_string()),
            move |request_body| {
                let captured_body_ref = captured_body_ref.clone();
                async move {
                    *captured_body_ref.lock().unwrap() = Some(request_body);
                    Ok::<Bytes, Error>(sample_non_stream_bytes())
                }
            },
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "chat.completion");
        assert_eq!(json["choices"][0]["message"]["content"], "Hello from upstream");
        let request_body = captured_body.lock().unwrap().clone().unwrap();
        assert!(request_body.contains("\"profileArn\":\"arn:aws:test\""));
        assert!(request_body.contains("\"modelId\":\"claude-sonnet-4.6\""));
    }

    #[tokio::test]
    async fn stream_request_returns_text_event_stream() {
        let response: Response = super::handle_stream_with_transport(
            ChatCompletionsRequest {
                model: "deepseek-chat".to_string(),
                messages: vec![user_message("Hello")],
                stream: true,
                max_tokens: None,
                temperature: None,
                top_p: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                reasoning_effort: None,
            },
            None,
            move |_| async move {
                let upstream = stream::iter(vec![Ok::<Bytes, Error>(sample_stream_bytes())]);
                Ok::<super::ByteStream, Error>(Box::pin(upstream))
            },
        )
        .await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(header::CONTENT_TYPE).unwrap(),
            "text/event-stream"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body_text = String::from_utf8(body.to_vec()).unwrap();
        assert!(body_text.contains("chat.completion.chunk"));
        assert!(body_text.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn provider_failure_maps_to_openai_error_envelope() {
        let response: Response = super::handle_non_stream_with_transport(
            ChatCompletionsRequest {
                model: "deepseek-chat".to_string(),
                messages: vec![user_message("Hello")],
                stream: false,
                max_tokens: None,
                temperature: None,
                top_p: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                reasoning_effort: None,
            },
            None,
            move |_| async move { Err::<Bytes, Error>(anyhow::anyhow!("upstream failed")) },
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "api_error");
        assert_eq!(json["error"]["message"], "upstream failed");
    }

    #[tokio::test]
    async fn auth_middleware_still_applies() {
        let app = create_router(AppState::new("test-key"));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": "deepseek-chat",
                "messages": [{ "role": "user", "content": "Hello" }]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    }
}
