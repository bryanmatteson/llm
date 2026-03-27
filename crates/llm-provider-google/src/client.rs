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
    Candidate, GenerateContentRequest, GenerateContentResponse, GenerationConfig,
    ModelListResponse, WireContent, WireFunctionCall, WireFunctionResponse, WirePart, WireTool,
};

/// Google Gemini-specific LLM client.
///
/// Wraps an HTTP client and an authenticated session to talk to the Google
/// Gemini generateContent and Models APIs.
pub struct GoogleClient {
    http: reqwest::Client,
    auth_session: AuthSession,
    base_url: String,
    model: ModelId,
}

impl GoogleClient {
    /// Create a new client.
    ///
    /// * `auth_session` - a previously authenticated session (API-key or OAuth).
    /// * `model`        - the model id to use (e.g. `"gemini-2.5-flash"`).
    /// * `base_url`     - optional API base URL override. If `None`, uses the
    ///   default Gemini API base.
    pub fn new(auth_session: AuthSession, model: ModelId, base_url: Option<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            auth_session,
            base_url: base_url.unwrap_or_else(|| API_BASE.to_string()),
            model,
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    /// Build the URL for a generateContent request, handling API key
    /// vs Bearer/OAuth auth.
    fn build_url(&self, endpoint: &str) -> String {
        let base = self.base_url.trim_end_matches('/');
        let url = format!("{base}/{endpoint}");

        // For API key auth, append the key as a query parameter.
        if matches!(
            self.auth_session.method,
            llm_auth::AuthMethod::ApiKey { .. }
        ) {
            format!("{url}?key={}", self.auth_session.tokens.access_token)
        } else {
            url
        }
    }

    /// Apply authentication to a request builder.
    fn apply_auth(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth_session.method {
            llm_auth::AuthMethod::ApiKey { .. } => {
                // API key is already in the URL query string.
                builder
            }
            llm_auth::AuthMethod::OAuth { .. } | llm_auth::AuthMethod::Bearer { .. } => {
                builder.bearer_auth(&self.auth_session.tokens.access_token)
            }
        }
    }

    /// Convert a canonical [`Message`] to a Gemini `WireContent`.
    pub(crate) fn message_to_wire(msg: &Message) -> WireContent {
        let role = match msg.role {
            Role::System => "user", // system handled separately
            Role::User => "user",
            Role::Assistant => "model",
            Role::Tool => "user", // tool results are sent as user messages in Gemini
        };

        let mut parts = Vec::new();

        // Check if this is a tool-result message.
        let has_tool_result = msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }));

        if has_tool_result {
            // Tool results are sent as functionResponse parts.
            for block in &msg.content {
                if let ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                } = block
                {
                    let response_value = match content {
                        serde_json::Value::String(text) => serde_json::from_str(text)
                            .unwrap_or_else(|_| serde_json::json!({"result": text})),
                        other => other.clone(),
                    };
                    parts.push(WirePart {
                        function_response: Some(WireFunctionResponse {
                            name: tool_use_id.clone(),
                            response: response_value,
                        }),
                        ..Default::default()
                    });
                }
            }
        } else {
            // Regular message: collect text and tool-use blocks.
            for block in &msg.content {
                match block {
                    ContentBlock::Text(text) => {
                        if !text.is_empty() {
                            parts.push(WirePart {
                                text: Some(text.clone()),
                                ..Default::default()
                            });
                        }
                    }
                    ContentBlock::ToolUse { name, input, .. } => {
                        parts.push(WirePart {
                            function_call: Some(WireFunctionCall {
                                name: name.clone(),
                                args: input.clone(),
                            }),
                            ..Default::default()
                        });
                    }
                    _ => {}
                }
            }
        }

        WireContent {
            role: Some(role.into()),
            parts,
        }
    }

    /// Convert a Gemini response candidate back into canonical [`Message`]s.
    fn wire_to_message(candidate: &Candidate) -> Message {
        let mut content = Vec::new();

        for part in &candidate.content.parts {
            if let Some(text) = &part.text {
                if !text.is_empty() {
                    content.push(ContentBlock::Text(text.clone()));
                }
            }
            if let Some(fc) = &part.function_call {
                content.push(ContentBlock::ToolUse {
                    // Gemini does not provide tool call IDs; synthesize one.
                    id: format!("gemini_call_{}", content.len()),
                    name: fc.name.clone(),
                    input: fc.args.clone(),
                });
            }
        }

        Message {
            role: Role::Assistant,
            content,
            metadata: Default::default(),
        }
    }

    /// Map a Gemini `finishReason` string to our canonical [`StopReason`].
    pub(crate) fn map_stop_reason(reason: Option<&str>) -> StopReason {
        match reason {
            Some("STOP") => StopReason::EndTurn,
            Some("MAX_TOKENS") => StopReason::MaxTokens,
            Some("SAFETY") | Some("RECITATION") => StopReason::Stop,
            // A missing finish_reason typically means the turn completed normally.
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

    /// Check for blocked prompts or safety-filtered responses.
    fn check_safety(response: &GenerateContentResponse) -> Result<()> {
        if let Some(feedback) = &response.prompt_feedback {
            if let Some(reason) = &feedback.block_reason {
                return Err(llm_core::FrameworkError::provider(
                    PROVIDER_ID.clone(),
                    format!("Gemini blocked the request: {reason}"),
                ));
            }
        }

        if let Some(candidate) = response.candidates.first() {
            if let Some(reason) = &candidate.finish_reason {
                if reason == "SAFETY" || reason == "RECITATION" {
                    return Err(llm_core::FrameworkError::provider(
                        PROVIDER_ID.clone(),
                        format!("Gemini filtered the response: {reason}"),
                    ));
                }
            }
        }

        Ok(())
    }
}

#[async_trait]
impl LlmProviderClient for GoogleClient {
    fn provider_id(&self) -> &ProviderId {
        &self.auth_session.provider_id
    }

    async fn send_turn(&self, request: &TurnRequest) -> Result<TurnResponse> {
        if self.auth_session.tokens.is_expired() {
            return Err(llm_core::FrameworkError::auth(
                "access token has expired; refresh or re-authenticate before making requests",
            ));
        }

        // ── Build contents ──────────────────────────────────────────
        let mut contents = Vec::new();

        for msg in &request.messages {
            contents.push(Self::message_to_wire(msg));
        }

        // ── Build system instruction ────────────────────────────────
        let system_instruction = request.system_prompt.as_ref().map(|sys| WireContent {
            role: None,
            parts: vec![WirePart {
                text: Some(sys.clone()),
                ..Default::default()
            }],
        });

        // ── Build tools ─────────────────────────────────────────────
        let tools: Vec<WireTool> = if request.tools.is_empty() {
            Vec::new()
        } else {
            // request.tools is already in the Google wire format (from GoogleToolFormat),
            // so we deserialize each one.
            request
                .tools
                .iter()
                .filter_map(|t| serde_json::from_value::<WireTool>(t.clone()).ok())
                .collect()
        };

        // ── Build generation config ─────────────────────────────────
        let generation_config = if request.max_tokens.is_some() || request.temperature.is_some() {
            Some(GenerationConfig {
                max_output_tokens: request.max_tokens,
                temperature: request.temperature,
            })
        } else {
            None
        };

        let model = request.model.as_ref().unwrap_or(&self.model).to_string();

        let body = GenerateContentRequest {
            contents,
            system_instruction,
            tools,
            generation_config,
        };

        // ── Send request ────────────────────────────────────────────
        let url = self.build_url(&format!("models/{model}:generateContent"));

        let builder = self
            .http
            .post(&url)
            .header("Content-Type", "application/json");

        let builder = self.apply_auth(builder);

        let resp = builder.json(&body).send().await.map_err(|e| {
            llm_core::FrameworkError::provider(PROVIDER_ID.clone(), format!("request failed: {e}"))
        })?;

        let resp = Self::check_response(resp, &PROVIDER_ID).await?;

        let response: GenerateContentResponse = resp.json().await.map_err(|e| {
            llm_core::FrameworkError::provider(
                PROVIDER_ID.clone(),
                format!("failed to parse response: {e}"),
            )
        })?;

        // ── Check for safety blocks ─────────────────────────────────
        Self::check_safety(&response)?;

        // ── Parse response ──────────────────────────────────────────
        let candidate = response.candidates.first().ok_or_else(|| {
            llm_core::FrameworkError::provider(
                PROVIDER_ID.clone(),
                "response contained no candidates",
            )
        })?;

        let message = Self::wire_to_message(candidate);
        let stop_reason = Self::map_stop_reason(candidate.finish_reason.as_deref());

        let usage = response
            .usage_metadata
            .map(|u| TokenUsage {
                input_tokens: u.prompt_token_count,
                output_tokens: u.candidates_token_count,
            })
            .unwrap_or_default();

        Ok(TurnResponse {
            messages: vec![message],
            stop_reason,
            model: ModelId::new(model),
            usage,
        })
    }

    async fn stream_turn(
        &self,
        _request: &TurnRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>> {
        Err(llm_core::FrameworkError::unsupported(
            "streaming is not supported for the Google Gemini provider (tool use requires non-streaming)",
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
        if self.auth_session.tokens.is_expired() {
            return Err(llm_core::FrameworkError::auth(
                "access token has expired; refresh or re-authenticate before making requests",
            ));
        }

        let url = self.build_url("models");

        let builder = self.http.get(&url);
        let builder = self.apply_auth(builder);

        let resp = builder.send().await.map_err(|e| {
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
            .models
            .into_iter()
            .map(|m| {
                // Strip the "models/" prefix from the resource name.
                let id = m.name.strip_prefix("models/").unwrap_or(&m.name);
                let display_name = m.display_name.unwrap_or_else(|| id.to_string());
                ModelDescriptor {
                    id: ModelId::new(id),
                    provider: PROVIDER_ID.clone(),
                    display_name,
                    context_window: m.input_token_limit,
                    capabilities: vec![],
                    metadata: Default::default(),
                }
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
        let wire = GoogleClient::message_to_wire(&msg);
        assert_eq!(wire.role.as_deref(), Some("user"));
        assert_eq!(wire.parts.len(), 1);
        assert_eq!(wire.parts[0].text.as_deref(), Some("Hello!"));
    }

    #[test]
    fn message_to_wire_assistant_text() {
        let msg = Message::assistant("Hi there!");
        let wire = GoogleClient::message_to_wire(&msg);
        assert_eq!(wire.role.as_deref(), Some("model"));
        assert_eq!(wire.parts.len(), 1);
        assert_eq!(wire.parts[0].text.as_deref(), Some("Hi there!"));
    }

    #[test]
    fn message_to_wire_assistant_with_tool_use() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("Let me check.".into()),
                ContentBlock::ToolUse {
                    id: "call_1".into(),
                    name: "search".into(),
                    input: json!({"q": "rust"}),
                },
            ],
            metadata: Default::default(),
        };

        let wire = GoogleClient::message_to_wire(&msg);
        assert_eq!(wire.role.as_deref(), Some("model"));
        assert_eq!(wire.parts.len(), 2);
        assert_eq!(wire.parts[0].text.as_deref(), Some("Let me check."));
        let fc = wire.parts[1].function_call.as_ref().unwrap();
        assert_eq!(fc.name, "search");
        assert_eq!(fc.args["q"], "rust");
    }

    #[test]
    fn message_to_wire_tool_result() {
        let msg = Message::tool_result("get_weather", "Sunny, 72F");
        let wire = GoogleClient::message_to_wire(&msg);
        assert_eq!(wire.role.as_deref(), Some("user"));
        assert_eq!(wire.parts.len(), 1);
        let fr = wire.parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.name, "get_weather");
    }

    #[test]
    fn message_to_wire_tool_result_json_content() {
        let msg = Message::tool_result("get_weather", r#"{"temp":72,"condition":"sunny"}"#);
        let wire = GoogleClient::message_to_wire(&msg);
        let fr = wire.parts[0].function_response.as_ref().unwrap();
        assert_eq!(fr.name, "get_weather");
        // JSON string content should be parsed as a JSON value.
        assert_eq!(fr.response["temp"], 72);
    }

    #[test]
    fn wire_to_message_text() {
        let candidate = Candidate {
            content: WireContent {
                role: Some("model".into()),
                parts: vec![WirePart {
                    text: Some("Hello there!".into()),
                    ..Default::default()
                }],
            },
            finish_reason: Some("STOP".into()),
        };

        let msg = GoogleClient::wire_to_message(&candidate);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text_content(), "Hello there!");
    }

    #[test]
    fn wire_to_message_with_function_call() {
        let candidate = Candidate {
            content: WireContent {
                role: Some("model".into()),
                parts: vec![
                    WirePart {
                        text: Some("Checking...".into()),
                        ..Default::default()
                    },
                    WirePart {
                        function_call: Some(WireFunctionCall {
                            name: "weather".into(),
                            args: json!({"city": "NYC"}),
                        }),
                        ..Default::default()
                    },
                ],
            },
            finish_reason: Some("STOP".into()),
        };

        let msg = GoogleClient::wire_to_message(&candidate);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 2);
        match &msg.content[0] {
            ContentBlock::Text(t) => assert_eq!(t, "Checking..."),
            other => panic!("expected Text, got {other:?}"),
        }
        match &msg.content[1] {
            ContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "weather");
                assert_eq!(input["city"], "NYC");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn map_stop_reason_values() {
        assert_eq!(
            GoogleClient::map_stop_reason(Some("STOP")),
            StopReason::EndTurn
        );
        assert_eq!(
            GoogleClient::map_stop_reason(Some("MAX_TOKENS")),
            StopReason::MaxTokens
        );
        assert_eq!(
            GoogleClient::map_stop_reason(Some("SAFETY")),
            StopReason::Stop
        );
        assert_eq!(
            GoogleClient::map_stop_reason(Some("RECITATION")),
            StopReason::Stop
        );
        assert_eq!(GoogleClient::map_stop_reason(None), StopReason::EndTurn);
        assert_eq!(
            GoogleClient::map_stop_reason(Some("UNKNOWN")),
            StopReason::EndTurn
        );
    }
}
