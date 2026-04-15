use std::pin::Pin;

use async_trait::async_trait;
use base64::Engine;
use reqwest::header::CONTENT_TYPE;
use serde_json::{Value, json};
use tokio_stream::{Stream, StreamExt};

use llm_auth::{AuthMethod, AuthSession};
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
/// Chat Completions and Models APIs.  When the session was established via
/// OAuth the client automatically targets the ChatGPT backend API and
/// includes the required `ChatGPT-Account-ID` header.
pub struct OpenAiClient {
    http: reqwest::Client,
    auth_session: AuthSession,
    base_url: String,
    model: ModelId,
    /// ChatGPT account ID extracted from the OAuth access-token JWT.
    /// When present, every request includes a `ChatGPT-Account-ID` header.
    chatgpt_account_id: Option<String>,
}

#[derive(Default)]
struct ResponsesStreamState {
    output: Vec<Value>,
    usage: TokenUsage,
    model: Option<ModelId>,
    incomplete_reason: Option<String>,
}

impl OpenAiClient {
    fn extract_chatgpt_account_id(access_token: &str) -> Option<String> {
        let payload_b64 = access_token.split('.').nth(1)?;
        let payload_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(payload_b64)
            .ok()?;
        let claims: Value = serde_json::from_slice(&payload_bytes).ok()?;
        claims
            .get("https://api.openai.com/auth")?
            .get("chatgpt_account_id")?
            .as_str()
            .map(str::to_string)
    }

    /// Create a new client.
    ///
    /// * `auth_session` – a previously authenticated session (API-key or OAuth).
    /// * `model`        – the model id to use (e.g. `"gpt-4o"`).
    /// * `base_url`     – the API base URL. When the session uses OAuth and no
    ///   explicit base URL is provided, the client automatically routes to
    ///   the ChatGPT backend API (`CHATGPT_API_BASE`).
    pub fn new(auth_session: AuthSession, model: ModelId, base_url: impl Into<String>) -> Self {
        let chatgpt_account_id = auth_session
            .metadata
            .get("chatgpt_account_id")
            .cloned()
            .or_else(|| Self::extract_chatgpt_account_id(&auth_session.tokens.access_token));
        Self {
            http: reqwest::Client::new(),
            auth_session,
            base_url: base_url.into(),
            model,
            chatgpt_account_id,
        }
    }

    /// Create a new client, automatically selecting the base URL from the
    /// auth session method.
    ///
    /// API key sessions use the public OpenAI API base. OAuth and bearer-token
    /// sessions route through the ChatGPT Codex backend.
    pub fn from_session(auth_session: AuthSession, model: ModelId) -> Self {
        let base_url = Self::default_base_url(&auth_session);
        Self::new(auth_session, model, base_url)
    }

    // ── Helpers ─────────────────────────────────────────────────────

    fn default_base_url(auth_session: &AuthSession) -> &'static str {
        match auth_session.method {
            AuthMethod::ApiKey { .. } => crate::descriptor::API_BASE,
            AuthMethod::OAuth { .. } | AuthMethod::Bearer { .. } => {
                crate::descriptor::CHATGPT_API_BASE
            }
        }
    }

    fn uses_responses_api(&self) -> bool {
        self.base_url.contains("chatgpt.com/backend-api")
    }

    /// Build the `Authorization` header value.
    fn auth_header(&self) -> String {
        format!("Bearer {}", self.auth_session.tokens.access_token)
    }

    fn tool_result_to_text(value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(text) => text.clone(),
            other => serde_json::to_string(other).unwrap_or_else(|_| other.to_string()),
        }
    }

    fn request_to_responses_input(request: &TurnRequest) -> Vec<Value> {
        let mut input = Vec::new();

        for msg in &request.messages {
            input.extend(Self::message_to_responses_input(msg));
        }

        input
    }

    fn message_to_responses_input(msg: &Message) -> Vec<Value> {
        let mut items = Vec::new();

        let role = match msg.role {
            Role::System => "developer",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "",
        };

        let text_items: Vec<Value> = msg
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) if !text.is_empty() => Some(json!({
                    "type": "input_text",
                    "text": text,
                })),
                _ => None,
            })
            .collect();

        if !text_items.is_empty() && msg.role != Role::Tool {
            items.push(json!({
                "type": "message",
                "role": role,
                "content": text_items,
            }));
        }

        for block in &msg.content {
            match block {
                ContentBlock::ToolUse { id, name, input } => {
                    items.push(json!({
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": serde_json::to_string(input).unwrap_or_default(),
                    }));
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                } => {
                    items.push(json!({
                        "type": "function_call_output",
                        "call_id": tool_use_id,
                        "output": Self::tool_result_to_text(content),
                    }));
                }
                ContentBlock::Text(_) => {}
            }
        }

        items
    }

    fn responses_tools(tools: &[Value]) -> Vec<Value> {
        tools
            .iter()
            .map(|tool| {
                if let Some(function) = tool.get("function") {
                    json!({
                        "type": "function",
                        "name": function.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "description": function.get("description").and_then(Value::as_str).unwrap_or_default(),
                        "parameters": function.get("parameters").cloned().unwrap_or_else(|| json!({})),
                    })
                } else {
                    tool.clone()
                }
            })
            .collect()
    }

    fn parse_responses_message(item: &Value) -> Vec<ContentBlock> {
        item.get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(
                |content| match content.get("type").and_then(Value::as_str) {
                    Some("output_text") | Some("input_text") => content
                        .get("text")
                        .and_then(Value::as_str)
                        .map(|text| ContentBlock::Text(text.to_string())),
                    _ => None,
                },
            )
            .collect()
    }

    fn parse_responses_output(value: &Value) -> Result<(Message, StopReason, TokenUsage, ModelId)> {
        let output = value
            .get("output")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                llm_core::FrameworkError::provider(
                    PROVIDER_ID.clone(),
                    "responses API reply did not include an output array",
                )
            })?;

        let mut content = Vec::new();
        let mut saw_tool_use = false;

        for item in output {
            match item.get("type").and_then(Value::as_str) {
                Some("message") => content.extend(Self::parse_responses_message(item)),
                Some("function_call") => {
                    let id = item
                        .get("call_id")
                        .or_else(|| item.get("id"))
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            llm_core::FrameworkError::provider(
                                PROVIDER_ID.clone(),
                                "responses API function_call item missing call_id",
                            )
                        })?;
                    let name = item.get("name").and_then(Value::as_str).ok_or_else(|| {
                        llm_core::FrameworkError::provider(
                            PROVIDER_ID.clone(),
                            "responses API function_call item missing name",
                        )
                    })?;
                    let arguments = item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    let input = serde_json::from_str(arguments).unwrap_or_default();
                    content.push(ContentBlock::ToolUse {
                        id: id.to_string(),
                        name: name.to_string(),
                        input,
                    });
                    saw_tool_use = true;
                }
                _ => {}
            }
        }

        let usage = value
            .get("usage")
            .and_then(Value::as_object)
            .map(|usage| TokenUsage {
                input_tokens: usage
                    .get("input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or_default(),
                output_tokens: usage
                    .get("output_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or_default(),
            })
            .unwrap_or_default();

        let model = value
            .get("model")
            .and_then(Value::as_str)
            .map(ModelId::new)
            .unwrap_or_else(|| ModelId::new("unknown"));

        let stop_reason = if saw_tool_use {
            StopReason::ToolUse
        } else if value
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
            == Some("max_output_tokens")
        {
            StopReason::MaxTokens
        } else {
            StopReason::EndTurn
        };

        Ok((
            Message {
                role: Role::Assistant,
                content,
                metadata: Default::default(),
            },
            stop_reason,
            usage,
            model,
        ))
    }

    fn upsert_responses_output_item(output: &mut Vec<Value>, item: Value) {
        let item_key = item
            .get("id")
            .or_else(|| item.get("call_id"))
            .and_then(Value::as_str);

        if let Some(item_key) = item_key
            && let Some(existing) = output.iter_mut().find(|existing| {
                existing
                    .get("id")
                    .or_else(|| existing.get("call_id"))
                    .and_then(Value::as_str)
                    == Some(item_key)
            })
        {
            *existing = item;
            return;
        }

        output.push(item);
    }

    fn handle_responses_stream_event(
        event_name: &str,
        data: &str,
        state: &mut ResponsesStreamState,
    ) -> Result<()> {
        let value: Value = serde_json::from_str(data).map_err(|e| {
            llm_core::FrameworkError::provider(
                PROVIDER_ID.clone(),
                format!("failed to parse OpenAI responses SSE payload: {e}"),
            )
        })?;

        match event_name {
            "error" | "response.failed" => {
                let message = value
                    .pointer("/error/message")
                    .or_else(|| value.pointer("/response/error/message"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown responses stream error");
                return Err(llm_core::FrameworkError::provider(
                    PROVIDER_ID.clone(),
                    format!("OpenAI responses stream error: {message}"),
                ));
            }
            "response.output_item.added" | "response.output_item.done" => {
                if let Some(item) = value.get("item").cloned() {
                    Self::upsert_responses_output_item(&mut state.output, item);
                }
            }
            "response.server_model" => {
                if let Some(model) = value.get("model").and_then(Value::as_str) {
                    state.model = Some(ModelId::new(model));
                }
            }
            "response.completed" => {
                if let Some(model) = value.pointer("/response/model").and_then(Value::as_str) {
                    state.model = Some(ModelId::new(model));
                }
                if let Some(reason) = value
                    .pointer("/response/incomplete_details/reason")
                    .and_then(Value::as_str)
                {
                    state.incomplete_reason = Some(reason.to_string());
                }
                if let Some(usage) = value.pointer("/response/usage").and_then(Value::as_object) {
                    state.usage = TokenUsage {
                        input_tokens: usage
                            .get("input_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or_default(),
                        output_tokens: usage
                            .get("output_tokens")
                            .and_then(Value::as_u64)
                            .unwrap_or_default(),
                    };
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn finalize_responses_stream(state: ResponsesStreamState, fallback_model: &ModelId) -> Value {
        let mut body = serde_json::Map::new();
        body.insert(
            "model".into(),
            Value::String(
                state
                    .model
                    .as_ref()
                    .map(|model| model.as_str().to_string())
                    .unwrap_or_else(|| fallback_model.as_str().to_string()),
            ),
        );
        body.insert("output".into(), Value::Array(state.output));
        body.insert(
            "usage".into(),
            json!({
                "input_tokens": state.usage.input_tokens,
                "output_tokens": state.usage.output_tokens,
            }),
        );
        if let Some(reason) = state.incomplete_reason {
            body.insert(
                "incomplete_details".into(),
                json!({
                    "reason": reason,
                }),
            );
        }
        Value::Object(body)
    }

    async fn parse_responses_sse(
        response: reqwest::Response,
        fallback_model: &ModelId,
    ) -> Result<Value> {
        let mut state = ResponsesStreamState::default();
        let mut bytes = response.bytes_stream();
        let mut pending = String::new();
        let mut event_name: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();

        while let Some(chunk) = bytes.next().await {
            let chunk = chunk.map_err(|e| {
                llm_core::FrameworkError::provider(
                    PROVIDER_ID.clone(),
                    format!("failed to read OpenAI responses stream: {e}"),
                )
            })?;

            pending.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(newline) = pending.find('\n') {
                let mut line = pending[..newline].to_string();
                pending.drain(..=newline);

                if line.ends_with('\r') {
                    line.pop();
                }

                if line.is_empty() {
                    if !data_lines.is_empty() {
                        let payload = data_lines.join("\n");
                        let derived_event =
                            serde_json::from_str::<Value>(&payload)
                                .ok()
                                .and_then(|value| {
                                    value
                                        .get("type")
                                        .and_then(Value::as_str)
                                        .map(str::to_string)
                                });
                        let event = event_name
                            .clone()
                            .or(derived_event)
                            .unwrap_or_else(|| "message".to_string());
                        Self::handle_responses_stream_event(&event, &payload, &mut state)?;
                    }
                    event_name = None;
                    data_lines.clear();
                    continue;
                }

                if let Some(rest) = line.strip_prefix("event:") {
                    event_name = Some(rest.trim_start().to_string());
                    continue;
                }
                if let Some(rest) = line.strip_prefix("data:") {
                    data_lines.push(rest.trim_start().to_string());
                }
            }
        }

        if !data_lines.is_empty() {
            let payload = data_lines.join("\n");
            let derived_event = serde_json::from_str::<Value>(&payload)
                .ok()
                .and_then(|value| {
                    value
                        .get("type")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                });
            let event = event_name
                .clone()
                .or(derived_event)
                .unwrap_or_else(|| "message".to_string());
            Self::handle_responses_stream_event(&event, &payload, &mut state)?;
        }

        Ok(Self::finalize_responses_stream(state, fallback_model))
    }

    #[cfg(test)]
    fn parse_responses_sse_text(document: &str, fallback_model: &ModelId) -> Result<Value> {
        let mut state = ResponsesStreamState::default();
        let mut event_name: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();

        for raw_line in document.lines() {
            let line = raw_line.trim_end_matches('\r');

            if line.is_empty() {
                if !data_lines.is_empty() {
                    let payload = data_lines.join("\n");
                    let derived_event = serde_json::from_str::<Value>(&payload)
                        .ok()
                        .and_then(|value| {
                            value
                                .get("type")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        });
                    let event = event_name
                        .clone()
                        .or(derived_event)
                        .unwrap_or_else(|| "message".to_string());
                    Self::handle_responses_stream_event(&event, &payload, &mut state)?;
                }
                event_name = None;
                data_lines.clear();
                continue;
            }

            if let Some(rest) = line.strip_prefix("event:") {
                event_name = Some(rest.trim_start().to_string());
                continue;
            }
            if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.trim_start().to_string());
            }
        }

        if !data_lines.is_empty() {
            let payload = data_lines.join("\n");
            let derived_event = serde_json::from_str::<Value>(&payload)
                .ok()
                .and_then(|value| {
                    value
                        .get("type")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                });
            let event = event_name
                .clone()
                .or(derived_event)
                .unwrap_or_else(|| "message".to_string());
            Self::handle_responses_stream_event(&event, &payload, &mut state)?;
        }

        Ok(Self::finalize_responses_stream(state, fallback_model))
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
                    ContentBlock::ToolResult { content, .. } => {
                        Some(Self::tool_result_to_text(content))
                    }
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

    #[cfg(test)]
    fn test_auth_session(method: AuthMethod) -> AuthSession {
        AuthSession {
            provider_id: PROVIDER_ID.clone(),
            method,
            tokens: llm_auth::TokenPair::new("token".into(), None, 3600),
            metadata: Default::default(),
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

        if self.uses_responses_api() {
            let model = request.model.as_ref().unwrap_or(&self.model).to_string();
            let mut body = serde_json::Map::new();
            body.insert("model".into(), Value::String(model));
            body.insert("store".into(), Value::Bool(false));
            body.insert("stream".into(), Value::Bool(true));
            if let Some(instructions) = request
                .system_prompt
                .as_deref()
                .map(str::trim)
                .filter(|text| !text.is_empty())
            {
                body.insert("instructions".into(), Value::String(instructions.to_string()));
            }
            body.insert(
                "input".into(),
                Value::Array(Self::request_to_responses_input(request)),
            );
            if let Some(temperature) = request.temperature {
                body.insert("temperature".into(), json!(temperature));
            }
            if let Some(max_tokens) = request.max_tokens {
                body.insert("max_output_tokens".into(), json!(max_tokens));
            }
            if !request.tools.is_empty() {
                body.insert(
                    "tools".into(),
                    Value::Array(Self::responses_tools(&request.tools)),
                );
            }

            let url = format!("{}/responses", self.base_url);

            let mut req = self
                .http
                .post(&url)
                .header("Authorization", self.auth_header())
                .header("Content-Type", "application/json");
            if let Some(account_id) = &self.chatgpt_account_id {
                req = req.header("ChatGPT-Account-ID", account_id);
            }

            let resp = req.json(&body).send().await.map_err(|e| {
                llm_core::FrameworkError::provider(
                    PROVIDER_ID.clone(),
                    format!("responses request failed: {e}"),
                )
            })?;

            let resp = Self::check_response(resp, &PROVIDER_ID).await?;
            let is_event_stream = resp
                .headers()
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok())
                .map(|content_type| content_type.contains("text/event-stream"))
                .unwrap_or(false);
            let response_value = if is_event_stream {
                Self::parse_responses_sse(resp, request.model.as_ref().unwrap_or(&self.model))
                    .await?
            } else {
                resp.json().await.map_err(|e| {
                    llm_core::FrameworkError::provider(
                        PROVIDER_ID.clone(),
                        format!("failed to parse responses API reply: {e}"),
                    )
                })?
            };

            let (message, stop_reason, usage, model) =
                Self::parse_responses_output(&response_value)?;

            return Ok(TurnResponse {
                messages: vec![message],
                stop_reason,
                model,
                usage,
            });
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

        let mut req = self
            .http
            .post(&url)
            .header("Authorization", self.auth_header())
            .header("Content-Type", "application/json");
        if let Some(account_id) = &self.chatgpt_account_id {
            req = req.header("ChatGPT-Account-ID", account_id);
        }
        let resp = req.json(&body).send().await.map_err(|e| {
            llm_core::FrameworkError::provider(PROVIDER_ID.clone(), format!("request failed: {e}"))
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

        let mut req = self
            .http
            .get(&url)
            .header("Authorization", self.auth_header());
        if let Some(account_id) = &self.chatgpt_account_id {
            req = req.header("ChatGPT-Account-ID", account_id);
        }
        let resp = req.send().await.map_err(|e| {
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

    #[test]
    fn from_session_uses_api_base_for_api_keys() {
        let client = OpenAiClient::from_session(
            OpenAiClient::test_auth_session(AuthMethod::ApiKey {
                masked: "sk-***".into(),
            }),
            ModelId::new("gpt-4o"),
        );
        assert_eq!(client.base_url, crate::descriptor::API_BASE);
    }

    #[test]
    fn from_session_uses_chatgpt_base_for_bearer_tokens() {
        let client = OpenAiClient::from_session(
            OpenAiClient::test_auth_session(AuthMethod::Bearer { expires_at: None }),
            ModelId::new("gpt-4o"),
        );
        assert_eq!(client.base_url, crate::descriptor::CHATGPT_API_BASE);
    }

    #[test]
    fn from_session_uses_chatgpt_base_for_oauth_sessions() {
        let client = OpenAiClient::from_session(
            OpenAiClient::test_auth_session(AuthMethod::OAuth {
                expires_at: chrono::Utc::now(),
            }),
            ModelId::new("gpt-4o"),
        );
        assert_eq!(client.base_url, crate::descriptor::CHATGPT_API_BASE);
    }

    #[test]
    fn request_to_responses_input_uses_typed_message_items() {
        let request = TurnRequest {
            system_prompt: Some("You are helpful.".into()),
            messages: vec![Message::user("Plan this repo.")],
            tools: vec![],
            provider_request: Default::default(),
            model: Some(ModelId::new("gpt-5")),
            max_tokens: Some(128),
            temperature: Some(0.2),
        };

        let input = OpenAiClient::request_to_responses_input(&request);
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["text"], "Plan this repo.");
    }

    #[test]
    fn responses_api_body_uses_top_level_instructions() {
        let request = TurnRequest {
            system_prompt: Some("You are helpful.".into()),
            messages: vec![Message::user("Plan this repo.")],
            tools: vec![],
            provider_request: Default::default(),
            model: Some(ModelId::new("gpt-5")),
            max_tokens: Some(128),
            temperature: Some(0.2),
        };

        let mut body = serde_json::Map::new();
        body.insert("store".into(), Value::Bool(false));
        body.insert("stream".into(), Value::Bool(true));
        if let Some(instructions) = request
            .system_prompt
            .as_deref()
            .map(str::trim)
            .filter(|text| !text.is_empty())
        {
            body.insert("instructions".into(), Value::String(instructions.to_string()));
        }
        body.insert(
            "input".into(),
            Value::Array(OpenAiClient::request_to_responses_input(&request)),
        );

        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(body["instructions"], "You are helpful.");
        assert_eq!(body["input"][0]["role"], "user");
    }

    #[test]
    fn parse_responses_sse_text_aggregates_stream_events() {
        let payload = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
            "event: response.server_model\n",
            "data: {\"type\":\"response.server_model\",\"model\":\"gpt-5.3-codex\"}\n\n",
            "event: response.output_item.added\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"id\":\"msg_1\",\"content\":[{\"type\":\"output_text\",\"text\":\"Draft\"}]}}\n\n",
            "event: response.output_item.done\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"id\":\"msg_1\",\"content\":[{\"type\":\"output_text\",\"text\":\"Final answer\"}]}}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":11,\"output_tokens\":7}}}\n\n",
        );

        let value = OpenAiClient::parse_responses_sse_text(payload, &ModelId::new("gpt-5"))
            .expect("parsed stream");
        let (message, stop_reason, usage, model) =
            OpenAiClient::parse_responses_output(&value).expect("parsed output");

        assert_eq!(model.as_str(), "gpt-5.3-codex");
        assert_eq!(message.text_content(), "Final answer");
        assert_eq!(stop_reason, StopReason::EndTurn);
        assert_eq!(usage.input_tokens, 11);
        assert_eq!(usage.output_tokens, 7);
    }

    #[test]
    fn parse_responses_output_extracts_text_and_tool_calls() {
        let response = json!({
            "model": "gpt-5.3-codex",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 7
            },
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "First,"
                        },
                        {
                            "type": "output_text",
                            "text": " second."
                        }
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "search",
                    "arguments": "{\"q\":\"router\"}"
                }
            ]
        });

        let (message, stop_reason, usage, model) =
            OpenAiClient::parse_responses_output(&response).expect("parsed response");

        assert_eq!(model.as_str(), "gpt-5.3-codex");
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(stop_reason, StopReason::ToolUse);
        assert_eq!(message.role, Role::Assistant);
        assert_eq!(message.text_content(), "First, second.");
        match &message.content[2] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "search");
                assert_eq!(input["q"], "router");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn new_extracts_chatgpt_account_id_from_token_when_metadata_is_empty() {
        let token = "header.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdF8xMjMifX0.sig";
        let auth_session = AuthSession {
            provider_id: PROVIDER_ID.clone(),
            method: AuthMethod::OAuth {
                expires_at: chrono::Utc::now(),
            },
            tokens: llm_auth::TokenPair::new(token.into(), None, 3600),
            metadata: Default::default(),
        };

        let client = OpenAiClient::new(
            auth_session,
            ModelId::new("gpt-5"),
            crate::descriptor::CHATGPT_API_BASE,
        );
        assert_eq!(client.chatgpt_account_id.as_deref(), Some("acct_123"));
    }
}
