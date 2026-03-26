use llm_core::{Metadata, ModelId, ProviderId};
use llm_tools::ToolPolicy;
use serde::{Deserialize, Serialize};

use crate::limits::SessionLimits;

/// Configuration for a conversation session.
///
/// Captures everything needed to set up a turn loop: which provider and model
/// to target, what system prompt to use, how tools should be gated, and
/// resource limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// The provider this session communicates with.
    pub provider_id: ProviderId,

    /// Optional model override.  When `None` the provider's default model is
    /// used.
    pub model: Option<ModelId>,

    /// Optional system prompt injected at the start of every turn request.
    pub system_prompt: Option<String>,

    /// Policy governing tool approval within this session.
    #[serde(default)]
    pub tool_policy: ToolPolicy,

    /// Resource limits for the turn loop.
    #[serde(default)]
    pub limits: SessionLimits,

    /// Arbitrary key-value metadata attached to this session.
    #[serde(default)]
    pub metadata: Metadata,
}
