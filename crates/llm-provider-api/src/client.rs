use std::pin::Pin;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

use serde_json::{Map, Value};

use llm_core::{Message, ModelDescriptor, ModelId, ProviderId, Result, StopReason, TokenUsage};

use crate::event::ProviderEvent;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRequest {
    pub system_prompt: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub provider_request: Map<String, Value>,
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

#[async_trait]
impl LlmProviderClient for Box<dyn LlmProviderClient> {
    fn provider_id(&self) -> &ProviderId {
        (**self).provider_id()
    }

    async fn send_turn(&self, request: &TurnRequest) -> Result<TurnResponse> {
        (**self).send_turn(request).await
    }

    async fn stream_turn(
        &self,
        request: &TurnRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>> {
        (**self).stream_turn(request).await
    }

    async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
        (**self).list_models().await
    }
}
