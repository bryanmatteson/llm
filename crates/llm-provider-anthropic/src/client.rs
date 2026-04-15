use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};

use llm_auth::AuthSession;
use llm_core::{
    ContentBlock, Message, Metadata, ModelDescriptor, ModelId, ProviderId, Result, Role,
    StopReason, TokenUsage,
};
use llm_provider_api::{LlmProviderClient, ProviderEvent, TurnRequest, TurnResponse};

use crate::descriptor::{API_BASE, PROVIDER_ID};
use crate::wire::{ContainerRequest, MessagesRequest, MessagesResponse, WireContent, WireMessage};

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

const RAW_CONTENT_METADATA_KEY: &str = "anthropic.raw_content_json";
const CONTAINER_ID_METADATA_KEY: &str = "anthropic.container_id";
const CONTAINER_EXPIRES_AT_METADATA_KEY: &str = "anthropic.container_expires_at";
const MESSAGE_CACHE_CONTROL_METADATA_KEY: &str = "anthropic.cache_control_json";
const CONTEXT_MANAGEMENT_METADATA_KEY: &str = "anthropic.context_management_json";
const TOOL_CALLER_TYPE_KEY: &str = "anthropic.caller_type";
const TOOL_CALLER_TOOL_ID_KEY: &str = "anthropic.caller_tool_id";
const ADVANCED_TOOL_USE_BETA: &str = "advanced-tool-use-2025-11-20";
const CONTEXT_MANAGEMENT_BETA: &str = "context-management-2025-06-27";
const OAUTH_BETA: &str = "oauth-2025-04-20";

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

    fn text_block(text: impl Into<String>) -> Value {
        json!({
            "type": "text",
            "text": text.into(),
        })
    }

    fn tool_use_block(id: &str, name: &str, input: Value) -> Value {
        json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": input,
        })
    }

    fn tool_result_block(tool_use_id: &str, content: &Value) -> Value {
        json!({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content.clone(),
        })
    }

    fn raw_blocks_from_metadata(msg: &Message) -> Option<Vec<Value>> {
        msg.metadata
            .get(RAW_CONTENT_METADATA_KEY)
            .and_then(|raw| serde_json::from_str(raw).ok())
    }

    fn tool_result_blocks(msg: &Message) -> Vec<Value> {
        let mut blocks: Vec<Value> = msg
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                } => Some(Self::tool_result_block(tool_use_id, content)),
                _ => None,
            })
            .collect();
        Self::apply_message_cache_control(msg, &mut blocks);
        blocks
    }

    fn apply_message_cache_control(msg: &Message, blocks: &mut [Value]) {
        let Some(raw) = msg.metadata.get(MESSAGE_CACHE_CONTROL_METADATA_KEY) else {
            return;
        };
        let Ok(cache_control) = serde_json::from_str::<Value>(raw) else {
            return;
        };
        let Some(last) = blocks.last_mut() else {
            return;
        };
        if let Some(obj) = last.as_object_mut() {
            obj.insert("cache_control".into(), cache_control);
        }
    }

    fn has_message_cache_control(msg: &Message) -> bool {
        msg.metadata
            .contains_key(MESSAGE_CACHE_CONTROL_METADATA_KEY)
    }

    fn beta_headers(&self, request: &TurnRequest) -> Vec<String> {
        let mut betas = Vec::new();

        // OAuth authentication requires the oauth beta header.
        if matches!(
            self.auth_session.method,
            llm_auth::AuthMethod::OAuth { .. } | llm_auth::AuthMethod::Bearer { .. }
        ) {
            betas.push(OAUTH_BETA.to_string());
        }

        if request.provider_request.contains_key("context_management") {
            betas.push(CONTEXT_MANAGEMENT_BETA.to_string());
        }

        let has_tool_search = request.tools.iter().any(|tool| {
            tool.get("type")
                .and_then(Value::as_str)
                .map(|tool_type| tool_type.starts_with("tool_search_tool_"))
                .unwrap_or(false)
                || tool
                    .get("defer_loading")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
        });
        if has_tool_search {
            betas.push(ADVANCED_TOOL_USE_BETA.to_string());
        }

        betas
    }

    fn extra_request_fields(request: &TurnRequest) -> serde_json::Map<String, Value> {
        let mut extra = request.provider_request.clone();
        extra.remove("system");
        extra.remove("cache_control");
        extra.remove("context_management");
        extra
    }

    /// Convert a canonical [`Message`] to the Anthropic wire format.
    fn message_to_wire(msg: &Message) -> WireMessage {
        let role = match msg.role {
            Role::System => "user", // system handled separately via `system` field
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "user", // tool results are sent as user messages in Anthropic
        };

        if let Some(blocks) = Self::raw_blocks_from_metadata(msg) {
            return WireMessage {
                role: role.into(),
                content: WireContent::Blocks(blocks),
            };
        }

        // Check if this is a tool-result message.
        let has_tool_result = msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }));

        if has_tool_result {
            return WireMessage {
                role: role.into(),
                content: WireContent::Blocks(Self::tool_result_blocks(msg)),
            };
        }

        // Check if the assistant message has tool-use blocks.
        let has_tool_use = msg
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }));

        if has_tool_use {
            let mut blocks = Vec::new();
            for block in &msg.content {
                match block {
                    ContentBlock::Text(text) => {
                        if !text.is_empty() {
                            blocks.push(Self::text_block(text));
                        }
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        blocks.push(Self::tool_use_block(id, name, input.clone()));
                    }
                    _ => {}
                }
            }
            Self::apply_message_cache_control(msg, &mut blocks);

            return WireMessage {
                role: role.into(),
                content: WireContent::Blocks(blocks),
            };
        }

        let text: String = msg
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        if Self::has_message_cache_control(msg) {
            let mut blocks = vec![Self::text_block(text)];
            Self::apply_message_cache_control(msg, &mut blocks);
            WireMessage {
                role: role.into(),
                content: WireContent::Blocks(blocks),
            }
        } else {
            WireMessage {
                role: role.into(),
                content: WireContent::Text(text),
            }
        }
    }

    fn messages_to_wire(messages: &[Message]) -> Vec<WireMessage> {
        let mut wire_messages = Vec::new();
        let mut pending_tool_results = Vec::new();

        for msg in messages {
            if msg.role == Role::Tool {
                pending_tool_results.extend(Self::tool_result_blocks(msg));
                continue;
            }

            if !pending_tool_results.is_empty() {
                wire_messages.push(WireMessage {
                    role: "user".into(),
                    content: WireContent::Blocks(std::mem::take(&mut pending_tool_results)),
                });
            }

            wire_messages.push(Self::message_to_wire(msg));
        }

        if !pending_tool_results.is_empty() {
            wire_messages.push(WireMessage {
                role: "user".into(),
                content: WireContent::Blocks(pending_tool_results),
            });
        }

        wire_messages
    }

    /// Convert an Anthropic wire response back into canonical [`Message`]s.
    fn wire_to_message(response: &MessagesResponse) -> Message {
        let mut content = Vec::new();
        let mut metadata = Metadata::new();

        if let Ok(raw_content) = serde_json::to_string(&response.content) {
            metadata.insert(RAW_CONTENT_METADATA_KEY.into(), raw_content);
        }

        if let Some(container) = &response.container {
            metadata.insert(CONTAINER_ID_METADATA_KEY.into(), container.id.clone());
            if let Some(expires_at) = &container.expires_at {
                metadata.insert(CONTAINER_EXPIRES_AT_METADATA_KEY.into(), expires_at.clone());
            }
        }
        if let Some(context_management) = &response.context_management {
            if let Ok(raw) = serde_json::to_string(context_management) {
                metadata.insert(CONTEXT_MANAGEMENT_METADATA_KEY.into(), raw);
            }
        }

        for block in &response.content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        if !text.is_empty() {
                            content.push(ContentBlock::Text(text.to_string()));
                        }
                    }
                }
                Some("tool_use") => {
                    let Some(id) = block.get("id").and_then(Value::as_str) else {
                        continue;
                    };
                    let Some(name) = block.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    let input = block
                        .get("input")
                        .cloned()
                        .unwrap_or(Value::Object(Default::default()));
                    content.push(ContentBlock::ToolUse {
                        id: id.to_string(),
                        name: name.to_string(),
                        input,
                    });
                }
                _ => {}
            }
        }

        Message {
            role: Role::Assistant,
            content,
            metadata,
        }
    }

    fn latest_container(messages: &[Message]) -> Option<ContainerRequest> {
        for msg in messages.iter().rev() {
            let Some(container_id) = msg.metadata.get(CONTAINER_ID_METADATA_KEY) else {
                continue;
            };

            if let Some(expires_at) = msg.metadata.get(CONTAINER_EXPIRES_AT_METADATA_KEY) {
                if let Ok(parsed) = DateTime::parse_from_rfc3339(expires_at) {
                    if parsed.with_timezone(&Utc) <= Utc::now() {
                        continue;
                    }
                }
            }

            return Some(ContainerRequest {
                id: Some(container_id.clone()),
            });
        }

        None
    }

    /// Map an Anthropic `stop_reason` string to our canonical [`StopReason`].
    fn map_stop_reason(reason: Option<&str>) -> StopReason {
        match reason {
            Some("end_turn") => StopReason::EndTurn,
            Some("tool_use") => StopReason::ToolUse,
            Some("pause_turn") => StopReason::PauseTurn,
            Some("max_tokens") | Some("model_context_window_exceeded") => StopReason::MaxTokens,
            Some("stop_sequence") | Some("refusal") => StopReason::Stop,
            None => StopReason::EndTurn,
            Some(_) => StopReason::EndTurn,
        }
    }

    fn apply_container_metadata(metadata: &mut Metadata, value: &Value) {
        let container = value
            .get("container")
            .or_else(|| value.get("message").and_then(|msg| msg.get("container")));

        let Some(container) = container else {
            return;
        };

        if let Some(id) = container.get("id").and_then(Value::as_str) {
            metadata.insert(CONTAINER_ID_METADATA_KEY.into(), id.to_string());
        }
        if let Some(expires_at) = container.get("expires_at").and_then(Value::as_str) {
            metadata.insert(
                CONTAINER_EXPIRES_AT_METADATA_KEY.into(),
                expires_at.to_string(),
            );
        }
    }

    async fn emit_event(
        tx: &mpsc::Sender<Result<ProviderEvent>>,
        event: ProviderEvent,
    ) -> Result<()> {
        tx.send(Ok(event)).await.map_err(|_| {
            llm_core::FrameworkError::provider(
                PROVIDER_ID.clone(),
                "stream receiver dropped unexpectedly",
            )
        })
    }

    async fn handle_stream_event(
        event_name: &str,
        data: &str,
        state: &mut AnthropicStreamState,
        tx: &mpsc::Sender<Result<ProviderEvent>>,
    ) -> Result<()> {
        let value: Value = serde_json::from_str(data).map_err(|e| {
            llm_core::FrameworkError::provider(
                PROVIDER_ID.clone(),
                format!("failed to parse Anthropic SSE payload: {e}"),
            )
        })?;

        match event_name {
            "ping" => {}
            "error" => {
                let message = value
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown stream error");
                return Err(llm_core::FrameworkError::provider(
                    PROVIDER_ID.clone(),
                    format!("Anthropic stream error: {message}"),
                ));
            }
            "message_start" => {
                if let Some(model) = value
                    .pointer("/message/model")
                    .and_then(Value::as_str)
                    .map(ModelId::new)
                {
                    state.model = Some(model);
                }
                if let Some(input_tokens) = value
                    .pointer("/message/usage/input_tokens")
                    .and_then(Value::as_u64)
                {
                    state.usage.input_tokens = input_tokens;
                }
                Self::apply_container_metadata(&mut state.metadata, &value);
            }
            "content_block_start" => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize;
                let Some(block) = value.get("content_block").cloned() else {
                    return Ok(());
                };
                let block_type = block
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                state.raw_blocks.insert(index, block.clone());

                if block_type == "tool_use" {
                    let Some(id) = block.get("id").and_then(Value::as_str) else {
                        return Ok(());
                    };
                    let Some(name) = block.get("name").and_then(Value::as_str) else {
                        return Ok(());
                    };
                    let mut metadata = Metadata::new();
                    if let Some(caller_type) = block.pointer("/caller/type").and_then(Value::as_str)
                    {
                        metadata.insert(TOOL_CALLER_TYPE_KEY.into(), caller_type.to_string());
                    }
                    if let Some(tool_id) = block.pointer("/caller/tool_id").and_then(Value::as_str)
                    {
                        metadata.insert(TOOL_CALLER_TOOL_ID_KEY.into(), tool_id.to_string());
                    }
                    Self::emit_event(
                        tx,
                        ProviderEvent::ToolCallStart {
                            id: id.to_string(),
                            name: name.to_string(),
                            metadata,
                        },
                    )
                    .await?;
                }
            }
            "content_block_delta" => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize;
                let Some(delta) = value.get("delta") else {
                    return Ok(());
                };

                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let Some(text) = delta.get("text").and_then(Value::as_str) else {
                            return Ok(());
                        };
                        if let Some(block) = state.raw_blocks.get_mut(&index) {
                            let current = block
                                .get("text")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            block["text"] = Value::String(format!("{current}{text}"));
                        }
                        Self::emit_event(
                            tx,
                            ProviderEvent::TextDelta {
                                text: text.to_string(),
                            },
                        )
                        .await?;
                    }
                    Some("input_json_delta") => {
                        let partial = delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        state.input_json.entry(index).or_default().push_str(partial);

                        if let Some(id) = state
                            .raw_blocks
                            .get(&index)
                            .and_then(|block| block.get("id"))
                            .and_then(Value::as_str)
                        {
                            Self::emit_event(
                                tx,
                                ProviderEvent::ToolCallDelta {
                                    id: id.to_string(),
                                    arguments_delta: partial.to_string(),
                                },
                            )
                            .await?;
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = value
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize;
                if let Some(raw_json) = state.input_json.remove(&index) {
                    if let Some(block) = state.raw_blocks.get_mut(&index) {
                        if let Ok(parsed) = serde_json::from_str::<Value>(&raw_json) {
                            block["input"] = parsed;
                        }
                    }

                    if let Some(id) = state
                        .raw_blocks
                        .get(&index)
                        .and_then(|block| block.get("id"))
                        .and_then(Value::as_str)
                    {
                        Self::emit_event(tx, ProviderEvent::ToolCallEnd { id: id.to_string() })
                            .await?;
                    }
                }
            }
            "message_delta" => {
                if let Some(stop_reason) =
                    value.pointer("/delta/stop_reason").and_then(Value::as_str)
                {
                    state.stop_reason = Some(Self::map_stop_reason(Some(stop_reason)));
                }
                if let Some(output_tokens) = value
                    .pointer("/usage/output_tokens")
                    .and_then(Value::as_u64)
                {
                    state.usage.output_tokens = output_tokens;
                }
                Self::apply_container_metadata(&mut state.metadata, &value);
            }
            "message_stop" => {
                state.completed = true;
                let mut blocks: Vec<(usize, Value)> = state
                    .raw_blocks
                    .iter()
                    .map(|(idx, block)| (*idx, block.clone()))
                    .collect();
                blocks.sort_by_key(|(idx, _)| *idx);
                let raw_content: Vec<Value> = blocks.into_iter().map(|(_, block)| block).collect();
                if let Ok(raw_json) = serde_json::to_string(&raw_content) {
                    state
                        .metadata
                        .insert(RAW_CONTENT_METADATA_KEY.into(), raw_json);
                }

                Self::emit_event(tx, ProviderEvent::UsageReported(state.usage.clone())).await?;
                Self::emit_event(
                    tx,
                    ProviderEvent::Done {
                        stop_reason: state.stop_reason.unwrap_or(StopReason::EndTurn),
                        model: state
                            .model
                            .clone()
                            .unwrap_or_else(|| ModelId::new("unknown")),
                        metadata: state.metadata.clone(),
                    },
                )
                .await?;
            }
            _ => {}
        }

        Ok(())
    }

    async fn pump_stream(
        response: reqwest::Response,
        tx: mpsc::Sender<Result<ProviderEvent>>,
    ) -> Result<()> {
        let mut state = AnthropicStreamState::default();
        let mut bytes = response.bytes_stream();
        let mut pending = String::new();
        let mut event_name: Option<String> = None;
        let mut data_lines: Vec<String> = Vec::new();

        while let Some(chunk) = bytes.next().await {
            let chunk = chunk.map_err(|e| {
                llm_core::FrameworkError::provider(
                    PROVIDER_ID.clone(),
                    format!("failed to read Anthropic stream: {e}"),
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
                        Self::handle_stream_event(&event, &payload, &mut state, &tx).await?;
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
            Self::handle_stream_event(&event, &payload, &mut state, &tx).await?;
        }

        if !state.completed {
            let mut blocks: Vec<(usize, Value)> = state
                .raw_blocks
                .iter()
                .map(|(idx, block)| (*idx, block.clone()))
                .collect();
            blocks.sort_by_key(|(idx, _)| *idx);
            let raw_content: Vec<Value> = blocks.into_iter().map(|(_, block)| block).collect();
            if let Ok(raw_json) = serde_json::to_string(&raw_content) {
                state
                    .metadata
                    .insert(RAW_CONTENT_METADATA_KEY.into(), raw_json);
            }
            Self::emit_event(&tx, ProviderEvent::UsageReported(state.usage.clone())).await?;
            Self::emit_event(
                &tx,
                ProviderEvent::Done {
                    stop_reason: state.stop_reason.unwrap_or(StopReason::EndTurn),
                    model: state.model.unwrap_or_else(|| ModelId::new("unknown")),
                    metadata: state.metadata,
                },
            )
            .await?;
        }

        Ok(())
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

#[derive(Default)]
struct AnthropicStreamState {
    raw_blocks: BTreeMap<usize, Value>,
    input_json: HashMap<usize, String>,
    usage: TokenUsage,
    stop_reason: Option<StopReason>,
    model: Option<ModelId>,
    metadata: Metadata,
    completed: bool,
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

        let wire_messages = Self::messages_to_wire(&request.messages);
        let model = request.model.as_ref().unwrap_or(&self.model).to_string();
        let max_tokens = request.max_tokens.unwrap_or(4096);

        let system = request
            .provider_request
            .get("system")
            .cloned()
            .or_else(|| request.system_prompt.clone().map(Value::String));
        let extra_body = Self::extra_request_fields(request);
        let body = MessagesRequest {
            model: model.clone(),
            max_tokens,
            temperature: request.temperature,
            system,
            container: Self::latest_container(&request.messages),
            cache_control: request.provider_request.get("cache_control").cloned(),
            context_management: request.provider_request.get("context_management").cloned(),
            messages: wire_messages,
            tools: request.tools.clone(),
            stream: false,
            extra_body,
        };

        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let builder = self
            .http
            .post(&url)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json");
        let betas = self.beta_headers(request);
        let builder = if betas.is_empty() {
            builder
        } else {
            builder.header("anthropic-beta", betas.join(","))
        };

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
        request: &TurnRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>> {
        if self.auth_session.tokens.is_expired() {
            return Err(llm_core::FrameworkError::auth(
                "access token has expired; refresh or re-authenticate before making requests",
            ));
        }

        let wire_messages = Self::messages_to_wire(&request.messages);
        let model = request.model.as_ref().unwrap_or(&self.model).to_string();
        let max_tokens = request.max_tokens.unwrap_or(4096);

        let system = request
            .provider_request
            .get("system")
            .cloned()
            .or_else(|| request.system_prompt.clone().map(Value::String));
        let extra_body = Self::extra_request_fields(request);
        let body = MessagesRequest {
            model,
            max_tokens,
            temperature: request.temperature,
            system,
            container: Self::latest_container(&request.messages),
            cache_control: request.provider_request.get("cache_control").cloned(),
            context_management: request.provider_request.get("context_management").cloned(),
            messages: wire_messages,
            tools: request.tools.clone(),
            stream: true,
            extra_body,
        };

        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));

        let builder = self
            .http
            .post(&url)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json");
        let betas = self.beta_headers(request);
        let builder = if betas.is_empty() {
            builder
        } else {
            builder.header("anthropic-beta", betas.join(","))
        };
        let builder = self.apply_auth(builder);

        let resp = builder.json(&body).send().await.map_err(|e| {
            llm_core::FrameworkError::provider(PROVIDER_ID.clone(), format!("request failed: {e}"))
        })?;
        let resp = Self::check_response(resp, &PROVIDER_ID).await?;

        let (tx, rx) = mpsc::channel(64);
        tokio::spawn(async move {
            if let Err(err) = Self::pump_stream(resp, tx.clone()).await {
                let _ = tx.send(Err(err)).await;
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
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
                assert_eq!(blocks[0]["type"], "text");
                assert_eq!(blocks[0]["text"], "Let me check.");
                assert_eq!(blocks[1]["type"], "tool_use");
                assert_eq!(blocks[1]["id"], "toolu_01");
                assert_eq!(blocks[1]["name"], "search");
                assert_eq!(blocks[1]["input"]["q"], "rust");
            }
            _ => panic!("expected Blocks content"),
        }
    }

    #[test]
    fn message_to_wire_tool_result() {
        let msg = Message::tool_result("toolu_01", "Sunny, 72F");
        let wire = AnthropicClient::message_to_wire(&msg);
        assert_eq!(wire.role, "user");
        match &wire.content {
            WireContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0]["type"], "tool_result");
                assert_eq!(blocks[0]["tool_use_id"], "toolu_01");
                assert_eq!(blocks[0]["content"], "Sunny, 72F");
            }
            _ => panic!("expected Blocks content"),
        }
    }

    #[test]
    fn message_to_wire_tool_result_preserves_structured_content() {
        let msg = Message::tool_result(
            "toolu_01",
            json!([
                {
                    "type": "tool_reference",
                    "tool_name": "get_weather"
                }
            ]),
        );
        let wire = AnthropicClient::message_to_wire(&msg);
        match &wire.content {
            WireContent::Blocks(blocks) => {
                assert_eq!(blocks[0]["content"][0]["type"], "tool_reference");
                assert_eq!(blocks[0]["content"][0]["tool_name"], "get_weather");
            }
            _ => panic!("expected Blocks content"),
        }
    }

    #[test]
    fn messages_to_wire_coalesces_adjacent_tool_results() {
        let messages = vec![
            Message::user("Hi"),
            Message::tool_result("toolu_01", "first"),
            Message::tool_result("toolu_02", "second"),
        ];

        let wire = AnthropicClient::messages_to_wire(&messages);
        assert_eq!(wire.len(), 2);
        match &wire[1].content {
            WireContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert_eq!(blocks[0]["tool_use_id"], "toolu_01");
                assert_eq!(blocks[1]["tool_use_id"], "toolu_02");
            }
            _ => panic!("expected tool result blocks"),
        }
    }

    #[test]
    fn wire_to_message_preserves_raw_content_and_container() {
        let response = MessagesResponse {
            id: "msg_01".into(),
            model: "claude-sonnet-4-20250514".into(),
            content: vec![
                json!({"type": "text", "text": "Checking..."}),
                json!({
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "weather",
                    "input": {"city": "NYC"},
                    "caller": {"type": "code_execution_20260120", "tool_id": "srvtoolu_01"}
                }),
            ],
            stop_reason: Some("tool_use".into()),
            usage: None,
            container: Some(crate::wire::ContainerResponse {
                id: "container_01".into(),
                expires_at: Some("2099-01-01T00:00:00Z".into()),
            }),
            context_management: None,
        };

        let msg = AnthropicClient::wire_to_message(&response);
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 2);
        assert!(msg.metadata.contains_key(RAW_CONTENT_METADATA_KEY));
        assert_eq!(
            msg.metadata
                .get(CONTAINER_ID_METADATA_KEY)
                .map(String::as_str),
            Some("container_01")
        );
    }

    #[test]
    fn message_to_wire_uses_raw_content_metadata() {
        let mut msg = Message::assistant("");
        msg.metadata.insert(
            RAW_CONTENT_METADATA_KEY.into(),
            serde_json::to_string(&vec![json!({
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {"query": "rust"}
            })])
            .unwrap(),
        );

        let wire = AnthropicClient::message_to_wire(&msg);
        match &wire.content {
            WireContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0]["type"], "server_tool_use");
            }
            _ => panic!("expected raw blocks"),
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
            AnthropicClient::map_stop_reason(Some("pause_turn")),
            StopReason::PauseTurn
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
    }
}
