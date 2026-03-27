use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::ids::{ModelId, ProviderId};
use crate::limits::SessionLimits;
use crate::metadata::Metadata;
use crate::policy::ToolPolicy;

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

    /// Provider-native tool descriptors that should be sent alongside any
    /// locally registered tools for this session.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub provider_tools: Vec<Value>,

    /// Provider-native request fields merged into each turn request.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub provider_request: Map<String, Value>,
}

impl SessionConfig {
    /// Create a minimal config for the given provider with all other fields
    /// set to their defaults.
    pub fn for_provider(provider_id: impl Into<ProviderId>) -> Self {
        Self {
            provider_id: provider_id.into(),
            model: None,
            system_prompt: None,
            tool_policy: ToolPolicy::default(),
            limits: SessionLimits::default(),
            metadata: Metadata::default(),
            provider_tools: Vec::new(),
            provider_request: Map::new(),
        }
    }
}
