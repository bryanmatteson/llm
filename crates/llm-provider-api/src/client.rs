use std::pin::Pin;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use llm_core::{Message, ModelDescriptor, ModelId, ProviderId, Result, StopReason, TokenUsage};

use crate::event::ProviderEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRequest {
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<serde_json::Value>,
    pub model: Option<ModelId>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnResponse {
    pub messages: Vec<Message>,
    pub stop_reason: StopReason,
    pub model: ModelId,
    pub usage: TokenUsage,
}

/// Async trait that every LLM provider client must implement.
#[async_trait]
pub trait LlmProviderClient: Send + Sync {
    /// Returns the provider identifier for this client.
    fn provider_id(&self) -> &ProviderId;

    /// Send a complete turn (non-streaming) and wait for the full response.
    async fn send_turn(&self, request: &TurnRequest) -> Result<TurnResponse>;

    /// Send a turn and receive a stream of incremental provider events.
    async fn stream_turn(
        &self,
        request: &TurnRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>>;

    /// List the models available from this provider.
    async fn list_models(&self) -> Result<Vec<ModelDescriptor>>;
}
