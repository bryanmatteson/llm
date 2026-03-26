use serde::{Deserialize, Serialize};

use llm_core::{Metadata, ModelId, ProviderId, SessionId};

/// Contextual information passed to a tool at execution time.
///
/// This gives the tool access to the identifiers for the current session,
/// provider, and model, as well as any arbitrary metadata the caller wishes
/// to forward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContext {
    /// The session in which this tool invocation occurs.
    pub session_id: SessionId,
    /// The provider that is driving the conversation.
    pub provider_id: ProviderId,
    /// The model that requested the tool call.
    pub model_id: ModelId,
    /// Arbitrary key-value metadata forwarded by the caller.
    pub metadata: Metadata,
}
