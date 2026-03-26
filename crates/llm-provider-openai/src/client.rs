use std::pin::Pin;

use async_trait::async_trait;
use tokio_stream::Stream;

use llm_auth::AuthSession;
use llm_core::{
    ContentBlock, Message, ModelDescriptor, ModelId, ProviderId, Result, Role, StopReason,
    TokenUsage,
};
use llm_provider_api::{LlmProviderClient, ProviderEvent, TurnRequest, TurnResponse};

use crate::descriptor::PROVIDER_ID;
use crate::wire::{
    ChatCompletionRequest, ChatCompletionResponse, ModelListResponse, WireFunctionCall,
    WireMessage, WireToolCall,
};

/// OpenAI-specific LLM client.
///
/// Wraps an HTTP client and an authenticated session to talk to the OpenAI
/// Chat Completions and Models APIs.
pub struct OpenAiClient {
    http: reqwest::Client,
    auth_session: AuthSession,
    base_url: String,
    model: ModelId,
}

impl OpenAiClient {
    /// Create a new client.
    ///
    /// * `auth_session` – a previously authenticated session (API-key or OAuth).
    /// * `model`        – the model id to use (e.g. `"gpt-4o"`).
    /// * `base_url`     – the API base URL (typically `https://api.openai.com/v1`).
    pub fn new(auth_session: AuthSession, model: ModelId, base_url: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            auth_session,
            base_url: base_url.into(),
            model,
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    /// Build the `Authorization` header value.
    fn auth_header(&self) -> String {
        format!("Bearer {}", self.auth_session.tokens.access_token)
    }

    /// Convert a canonical [`Message`] to the OpenAI wire format.
    fn message_to_wire(msg: &Message) -> WireMessage {
        let role = match msg.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };

        // Collect plain text content.
        let text: String = msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        // Collect tool-use blocks (assistant requesting tool calls).
        let tool_calls: Vec<WireToolCall> = msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::ToolUse { id, name, input } => Some(WireToolCall {
                    id: id.clone(),
                    type_field: "function".into(),
                    function: WireFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                }),
                _ => None,
            })
            .collect();

        // Extract tool_call_id for tool-result messages.
        let tool_call_id = msg.content.iter().find_map(|b| match b {
            ContentBlock::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
            _ => None,
        });

        // For tool result messages, the content is the result text.
        let content_text = if msg.role == Role::Tool {
            msg.content
                .iter()
                .find_map(|b| match b {
                    ContentBlock::ToolResult { content, .. } => Some(content.clone()),
                    _ => None,
                })
                .or_else(|| {
                    if text.is_empty() {
                        None
                    } else {
                        Some(text.clone())
                    }
                })
        } else if text.is_empty() && !tool_calls.is_empty() {
            None
        } else {
            Some(text)
        };

        WireMessage {
            role: role.into(),
            content: content_text,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id,
            name: None,
        }
    }

    /// Convert an OpenAI wire message back into a canonical [`Message`].
    fn wire_to_message(wire: &WireMessage) -> Message {
        let role = match wire.role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "tool" => Role::Tool,
            _ => Role::Assistant,
        };

        let mut content = Vec::new();

        // Text content.
        if let Some(text) = &wire.content {
            if !text.is_empty() {
                content.push(ContentBlock::Text(text.clone()));
            }
        }

        // Tool calls.
        if let Some(tool_calls) = &wire.tool_calls {
            for tc in tool_calls {
                let input: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                content.push(ContentBlock::ToolUse {
                    id: tc.id.clone(),
                    name: tc.function.name.clone(),
                    input,
                });
            }
        }

        Message {
            role,
            content,
            metadata: Default::default(),
        }
    }

    /// Map an OpenAI `finish_reason` string to our canonical [`StopReason`].
    fn map_stop_reason(reason: Option<&str>) -> StopReason {
        match reason {
            Some("stop") => StopReason::EndTurn,
            Some("tool_calls") => StopReason::ToolUse,
            Some("length") => StopReason::MaxTokens,
            // A missing finish_reason in a non-streaming response typically
            // means the turn completed normally.
            None => StopReason::EndTurn,
            // Unknown reason string — treat as end-of-turn rather than
            // silently dropping content.
            Some(_) => StopReason::EndTurn,
        }
    }

    /// Read the response body and, if the status is not 2xx, return a
    /// `FrameworkError::Provider`.
    async fn check_response(
        resp: reqwest::Response,
        provider: &ProviderId,
    ) -> Result<reqwest::Response> {
        if resp.status().is_success() {
            return Ok(resp);
        }

        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        Err(llm_core::FrameworkError::provider(
            provider.clone(),
            format!("HTTP {status}: {body}"),
        ))
    }
}

#[async_trait]
impl LlmProviderClient for OpenAiClient {
    fn provider_id(&self) -> &ProviderId {
        &self.auth_session.provider_id
    }

    async fn send_turn(&self, request: &TurnRequest) -> Result<TurnResponse> {
        if self.auth_session.tokens.is_expired() {
            return Err(llm_core::FrameworkError::auth(
                "access token has expired; refresh or re-authenticate before making requests",
            ));
        }

        // ── Build messages ──────────────────────────────────────────
        let mut wire_messages = Vec::new();

        // Prepend system prompt as a system message if provided.
        if let Some(sys) = &request.system_prompt {
            wire_messages.push(WireMessage {
                role: "system".into(),
                content: Some(sys.clone()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }

        for msg in &request.messages {
            wire_messages.push(Self::message_to_wire(msg));
        }

        // ── Build request body ──────────────────────────────────────
        let model = request.model.as_ref().unwrap_or(&self.model).to_string();

        let tools = if request.tools.is_empty() {
            None
        } else {
            Some(request.tools.clone())
        };

        let body = ChatCompletionRequest {
            model: model.clone(),
            messages: wire_messages,
            tools,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
        };

        // ── Send request ────────────────────────────────────────────
        let url = format!("{}/chat/completions", self.base_url);

        let resp = self
            .http
            .post(&url)
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                llm_core::FrameworkError::provider(
                    PROVIDER_ID.clone(),
                    format!("request failed: {e}"),
                )
            })?;

        let resp = Self::check_response(resp, &PROVIDER_ID).await?;

        let completion: ChatCompletionResponse = resp.json().await.map_err(|e| {
            llm_core::FrameworkError::provider(
                PROVIDER_ID.clone(),
                format!("failed to parse response: {e}"),
            )
        })?;

        // ── Parse response ──────────────────────────────────────────
        let choice = completion.choices.first().ok_or_else(|| {
            llm_core::FrameworkError::provider(PROVIDER_ID.clone(), "response contained no choices")
        })?;

        let message = Self::wire_to_message(&choice.message);
        let stop_reason = Self::map_stop_reason(choice.finish_reason.as_deref());

        let usage = completion
            .usage
            .map(|u| TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
            })
            .unwrap_or_default();

        Ok(TurnResponse {
            messages: vec![message],
            stop_reason,
            model: ModelId::new(completion.model),
            usage,
        })
    }

    async fn stream_turn(
        &self,
        _request: &TurnRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>> {
        Err(llm_core::FrameworkError::unsupported(
            "streaming is not yet implemented for the OpenAI provider",
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
        if self.auth_session.tokens.is_expired() {
            return Err(llm_core::FrameworkError::auth(
                "access token has expired; refresh or re-authenticate before making requests",
            ));
        }

        let url = format!("{}/models", self.base_url);

        let resp = self
            .http
            .get(&url)
            .header("Authorization", self.auth_header())
            .send()
            .await
            .map_err(|e| {
                llm_core::FrameworkError::provider(
                    PROVIDER_ID.clone(),
                    format!("list models request failed: {e}"),
                )
            })?;

        let resp = Self::check_response(resp, &PROVIDER_ID).await?;

        let list: ModelListResponse = resp.json().await.map_err(|e| {
            llm_core::FrameworkError::provider(
                PROVIDER_ID.clone(),
                format!("failed to parse model list: {e}"),
            )
        })?;

        Ok(list
            .data
            .into_iter()
            .map(|m| ModelDescriptor {
                id: ModelId::new(m.id.as_str()),
                provider: PROVIDER_ID.clone(),
                display_name: m.id.clone(),
                context_window: None,
                capabilities: vec![],
                metadata: Default::default(),
            })
            .collect())
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use llm_core::ContentBlock;
    use serde_json::json;

    use super::*;

    #[test]
    fn message_to_wire_user() {
        let msg = Message::user("Hello!");
        let wire = OpenAiClient::message_to_wire(&msg);
        assert_eq!(wire.role, "user");
        assert_eq!(wire.content.as_deref(), Some("Hello!"));
        assert!(wire.tool_calls.is_none());
    }

    #[test]
    fn message_to_wire_assistant_with_tool_use() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call_1".into(),
                name: "search".into(),
                input: json!({"q": "rust"}),
            }],
            metadata: Default::default(),
        };

        let wire = OpenAiClient::message_to_wire(&msg);
        assert_eq!(wire.role, "assistant");
        assert!(wire.content.is_none()); // no text content
        let tcs = wire.tool_calls.unwrap();
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].id, "call_1");
        assert_eq!(tcs[0].function.name, "search");
    }

    #[test]
    fn message_to_wire_tool_result() {
        let msg = Message::tool_result("call_1", "result text");
        let wire = OpenAiClient::message_to_wire(&msg);
        assert_eq!(wire.role, "tool");
        assert_eq!(wire.content.as_deref(), Some("result text"));
        assert_eq!(wire.tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn wire_to_message_text() {
        let wire = WireMessage {
            role: "assistant".into(),
            content: Some("Hi there!".into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };

        let msg = OpenAiClient::wire_to_message(&wire);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text_content(), "Hi there!");
    }

    #[test]
    fn wire_to_message_tool_calls() {
        let wire = WireMessage {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(vec![WireToolCall {
                id: "call_99".into(),
                type_field: "function".into(),
                function: WireFunctionCall {
                    name: "weather".into(),
                    arguments: r#"{"city":"NYC"}"#.into(),
                },
            }]),
            tool_call_id: None,
            name: None,
        };

        let msg = OpenAiClient::wire_to_message(&wire);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_99");
                assert_eq!(name, "weather");
                assert_eq!(input["city"], "NYC");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn map_stop_reason_values() {
        assert_eq!(
            OpenAiClient::map_stop_reason(Some("stop")),
            StopReason::EndTurn
        );
        assert_eq!(
            OpenAiClient::map_stop_reason(Some("tool_calls")),
            StopReason::ToolUse
        );
        assert_eq!(
            OpenAiClient::map_stop_reason(Some("length")),
            StopReason::MaxTokens
        );
        assert_eq!(OpenAiClient::map_stop_reason(None), StopReason::EndTurn);
        assert_eq!(
            OpenAiClient::map_stop_reason(Some("unknown")),
            StopReason::EndTurn
        );
    }
}
