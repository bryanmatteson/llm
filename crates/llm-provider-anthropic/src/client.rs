use std::pin::Pin;

use async_trait::async_trait;
use tokio_stream::Stream;

use llm_auth::AuthSession;
use llm_core::{
    ContentBlock, Message, ModelDescriptor, ModelId, ProviderId, Result, Role, StopReason,
    TokenUsage,
};
use llm_provider_api::{LlmProviderClient, ProviderEvent, TurnRequest, TurnResponse};

use crate::descriptor::{API_BASE, PROVIDER_ID};
use crate::wire::{
    ContentBlock as WireContentBlock, MessagesRequest, MessagesResponse, WireContent, WireMessage,
};

/// Anthropic-specific LLM client.
///
/// Wraps an HTTP client and an authenticated session to talk to the Anthropic
/// Messages API.
pub struct AnthropicClient {
    http: reqwest::Client,
    auth_session: AuthSession,
    base_url: String,
    model: ModelId,
}

impl AnthropicClient {
    /// Create a new client.
    ///
    /// * `auth_session` - a previously authenticated session (API-key or OAuth).
    /// * `model`        - the model id to use (e.g. `"claude-sonnet-4-20250514"`).
    /// * `base_url`     - optional API base URL override. If `None`, uses the
    ///   default Anthropic API base.
    pub fn new(auth_session: AuthSession, model: ModelId, base_url: Option<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            auth_session,
            base_url: base_url.unwrap_or_else(|| API_BASE.to_string()),
            model,
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    /// Build auth headers based on the session's authentication method.
    ///
    /// - API key auth: `x-api-key` header
    /// - OAuth / Bearer: `Authorization: Bearer` header
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        use llm_auth::AuthMethod;
        match &self.auth_session.method {
            AuthMethod::ApiKey { .. } => {
                builder.header("x-api-key", &self.auth_session.tokens.access_token)
            }
            AuthMethod::OAuth { .. } | AuthMethod::Bearer { .. } => {
                builder.bearer_auth(&self.auth_session.tokens.access_token)
            }
        }
    }

    /// Convert a canonical [`Message`] to the Anthropic wire format.
    fn message_to_wire(msg: &Message) -> WireMessage {
        let role = match msg.role {
            Role::System => "user", // system handled separately via `system` field
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "user", // tool results are sent as user messages in Anthropic
        };

        // Check if this is a tool-result message.
        let has_tool_result = msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }));

        if has_tool_result {
            // Tool results are sent as user messages with tool_result content blocks.
            let blocks: Vec<WireContentBlock> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                    } => Some(WireContentBlock::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: serde_json::Value::String(content.clone()),
                    }),
                    _ => None,
                })
                .collect();

            return WireMessage {
                role: role.into(),
                content: WireContent::Blocks(blocks),
            };
        }

        // Check if the assistant message has tool-use blocks.
        let has_tool_use = msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

        if has_tool_use {
            // Assistant messages with tool_use need content blocks.
            let mut blocks = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text(text) => {
                        if !text.is_empty() {
                            blocks.push(WireContentBlock::Text { text: text.clone() });
                        }
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        blocks.push(WireContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        });
                    }
                    _ => {}
                }
            }

            return WireMessage {
                role: role.into(),
                content: WireContent::Blocks(blocks),
            };
        }

        // Plain text message.
        let text: String = msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        WireMessage {
            role: role.into(),
            content: WireContent::Text(text),
        }
    }

    /// Convert an Anthropic wire response back into canonical [`Message`]s.
    fn wire_to_message(response: &MessagesResponse) -> Message {
        let mut content = Vec::new();

        for block in &response.content {
            match block {
                WireContentBlock::Text { text } => {
                    if !text.is_empty() {
                        content.push(ContentBlock::Text(text.clone()));
                    }
                }
                WireContentBlock::ToolUse { id, name, input } => {
                    content.push(ContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    });
                }
                WireContentBlock::ToolResult { .. } => {
                    // ToolResult blocks shouldn't appear in responses, but handle gracefully.
                }
            }
        }

        Message {
            role: Role::Assistant,
            content,
            metadata: Default::default(),
        }
    }

    /// Map an Anthropic `stop_reason` string to our canonical [`StopReason`].
    fn map_stop_reason(reason: Option<&str>) -> StopReason {
        match reason {
            Some("end_turn") => StopReason::EndTurn,
            Some("tool_use") => StopReason::ToolUse,
            Some("max_tokens") => StopReason::MaxTokens,
            Some("stop_sequence") => StopReason::Stop,
            // A missing stop_reason typically means the turn completed normally.
            None => StopReason::EndTurn,
            // Unknown reason string -- treat as end-of-turn.
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
impl LlmProviderClient for AnthropicClient {
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
        let wire_messages: Vec<WireMessage> =
            request.messages.iter().map(Self::message_to_wire).collect();

        // ── Build request body ──────────────────────────────────────
        let model = request.model.as_ref().unwrap_or(&self.model).to_string();

        let max_tokens = request.max_tokens.unwrap_or(4096);

        let body = MessagesRequest {
            model: model.clone(),
            max_tokens,
            system: request.system_prompt.clone(),
            messages: wire_messages,
            tools: request.tools.clone(),
            stream: false,
        };

        // ── Send request ────────────────────────────────────────────
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let builder = self
            .http
            .post(&url)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json");

        let builder = self.apply_auth(builder);

        let resp = builder.json(&body).send().await.map_err(|e| {
            llm_core::FrameworkError::provider(PROVIDER_ID.clone(), format!("request failed: {e}"))
        })?;

        let resp = Self::check_response(resp, &PROVIDER_ID).await?;

        let response: MessagesResponse = resp.json().await.map_err(|e| {
            llm_core::FrameworkError::provider(
                PROVIDER_ID.clone(),
                format!("failed to parse response: {e}"),
            )
        })?;

        // ── Parse response ──────────────────────────────────────────
        let message = Self::wire_to_message(&response);
        let stop_reason = Self::map_stop_reason(response.stop_reason.as_deref());

        let usage = response
            .usage
            .map(|u| TokenUsage {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
            })
            .unwrap_or_default();

        Ok(TurnResponse {
            messages: vec![message],
            stop_reason,
            model: ModelId::new(response.model),
            usage,
        })
    }

    async fn stream_turn(
        &self,
        _request: &TurnRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>> {
        Err(llm_core::FrameworkError::unsupported(
            "streaming is not yet implemented for the Anthropic provider",
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
        // Anthropic doesn't have a public models list endpoint, so we return
        // a hardcoded list of known models.
        let models = vec![
            ("claude-sonnet-4-20250514", "Claude Sonnet 4", Some(200_000)),
            (
                "claude-haiku-3-5-20241022",
                "Claude 3.5 Haiku",
                Some(200_000),
            ),
            ("claude-opus-4-20250514", "Claude Opus 4", Some(200_000)),
        ];

        Ok(models
            .into_iter()
            .map(|(id, name, ctx)| ModelDescriptor {
                id: ModelId::new(id),
                provider: PROVIDER_ID.clone(),
                display_name: name.to_string(),
                context_window: ctx,
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
        let wire = AnthropicClient::message_to_wire(&msg);
        assert_eq!(wire.role, "user");
        match &wire.content {
            WireContent::Text(t) => assert_eq!(t, "Hello!"),
            _ => panic!("expected Text content"),
        }
    }

    #[test]
    fn message_to_wire_assistant_with_tool_use() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("Let me check.".into()),
                ContentBlock::ToolUse {
                    id: "toolu_01".into(),
                    name: "search".into(),
                    input: json!({"q": "rust"}),
                },
            ],
            metadata: Default::default(),
        };

        let wire = AnthropicClient::message_to_wire(&msg);
        assert_eq!(wire.role, "assistant");
        match &wire.content {
            WireContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    WireContentBlock::Text { text } => assert_eq!(text, "Let me check."),
                    other => panic!("expected Text block, got {other:?}"),
                }
                match &blocks[1] {
                    WireContentBlock::ToolUse { id, name, input } => {
                        assert_eq!(id, "toolu_01");
                        assert_eq!(name, "search");
                        assert_eq!(input["q"], "rust");
                    }
                    other => panic!("expected ToolUse block, got {other:?}"),
                }
            }
            _ => panic!("expected Blocks content"),
        }
    }

    #[test]
    fn message_to_wire_tool_result() {
        let msg = Message::tool_result("toolu_01", "Sunny, 72F");
        let wire = AnthropicClient::message_to_wire(&msg);
        assert_eq!(wire.role, "user"); // tool results are user messages in Anthropic
        match &wire.content {
            WireContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    WireContentBlock::ToolResult {
                        tool_use_id,
                        content,
                    } => {
                        assert_eq!(tool_use_id, "toolu_01");
                        assert_eq!(content, &json!("Sunny, 72F"));
                    }
                    other => panic!("expected ToolResult block, got {other:?}"),
                }
            }
            _ => panic!("expected Blocks content"),
        }
    }

    #[test]
    fn wire_to_message_text() {
        let response = MessagesResponse {
            id: "msg_01".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![WireContentBlock::Text {
                text: "Hi there!".into(),
            }],
            stop_reason: Some("end_turn".into()),
            usage: None,
        };

        let msg = AnthropicClient::wire_to_message(&response);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text_content(), "Hi there!");
    }

    #[test]
    fn wire_to_message_with_tool_use() {
        let response = MessagesResponse {
            id: "msg_02".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![
                WireContentBlock::Text {
                    text: "Checking...".into(),
                },
                WireContentBlock::ToolUse {
                    id: "toolu_01".into(),
                    name: "weather".into(),
                    input: json!({"city": "NYC"}),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: None,
        };

        let msg = AnthropicClient::wire_to_message(&response);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 2);
        match &msg.content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "weather");
                assert_eq!(input["city"], "NYC");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn map_stop_reason_values() {
        assert_eq!(
            AnthropicClient::map_stop_reason(Some("end_turn")),
            StopReason::EndTurn
        );
        assert_eq!(
            AnthropicClient::map_stop_reason(Some("tool_use")),
            StopReason::ToolUse
        );
        assert_eq!(
            AnthropicClient::map_stop_reason(Some("max_tokens")),
            StopReason::MaxTokens
        );
        assert_eq!(
            AnthropicClient::map_stop_reason(Some("stop_sequence")),
            StopReason::Stop
        );
        assert_eq!(AnthropicClient::map_stop_reason(None), StopReason::EndTurn);
        assert_eq!(
            AnthropicClient::map_stop_reason(Some("unknown")),
            StopReason::EndTurn
        );
    }

    #[test]
    fn message_to_wire_assistant_text_only() {
        let msg = Message::assistant("Just a text reply.");
        let wire = AnthropicClient::message_to_wire(&msg);
        assert_eq!(wire.role, "assistant");
        match &wire.content {
            WireContent::Text(t) => assert_eq!(t, "Just a text reply."),
            _ => panic!("expected Text content for plain assistant message"),
        }
    }
}
