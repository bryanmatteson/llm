use std::sync::LazyLock;

use llm_core::{
    Metadata, ModelId, ProviderCapability, ProviderDescriptor, ProviderId,
};

/// Provider identifier used across the framework.
pub static PROVIDER_ID: LazyLock<ProviderId> = LazyLock::new(|| ProviderId::new("anthropic"));

/// The default model to use when none is specified.
pub static DEFAULT_MODEL: LazyLock<ModelId> =
    LazyLock::new(|| ModelId::new("claude-sonnet-4-20250514"));

/// Base URL for the Anthropic REST API.
pub const API_BASE: &str = "https://api.anthropic.com/v1";

/// The OAuth client-id registered for this application with Anthropic
/// (same as Claude Code CLI).
pub const ANTHROPIC_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";

/// Build the canonical [`ProviderDescriptor`] for Anthropic.
pub fn provider_descriptor() -> ProviderDescriptor {
    ProviderDescriptor {
        id: PROVIDER_ID.clone(),
        display_name: "Anthropic".to_owned(),
        default_model: DEFAULT_MODEL.clone(),
        capabilities: vec![
            ProviderCapability::ApiKeyAuth,
            ProviderCapability::OAuth,
            ProviderCapability::ToolCalling,
            ProviderCapability::Streaming,
            ProviderCapability::SystemPrompt,
        ],
        metadata: Metadata::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_id_is_anthropic() {
        assert_eq!(PROVIDER_ID.as_str(), "anthropic");
    }

    #[test]
    fn default_model_is_claude_sonnet() {
        assert_eq!(DEFAULT_MODEL.as_str(), "claude-sonnet-4-20250514");
    }

    #[test]
    fn descriptor_has_expected_capabilities() {
        let desc = provider_descriptor();
        assert_eq!(desc.id.as_str(), "anthropic");
        assert_eq!(desc.display_name, "Anthropic");
        assert!(desc.capabilities.contains(&ProviderCapability::ToolCalling));
        assert!(desc.capabilities.contains(&ProviderCapability::OAuth));
        assert!(desc.capabilities.contains(&ProviderCapability::ApiKeyAuth));
        assert!(desc.capabilities.contains(&ProviderCapability::Streaming));
        assert!(desc.capabilities.contains(&ProviderCapability::SystemPrompt));
    }
}
