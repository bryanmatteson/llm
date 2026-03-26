use std::sync::LazyLock;

use llm_core::{Metadata, ModelId, ProviderCapability, ProviderDescriptor, ProviderId};

/// Provider identifier used across the framework.
pub static PROVIDER_ID: LazyLock<ProviderId> = LazyLock::new(|| ProviderId::new("openai"));

/// The default model to use when none is specified.
pub static DEFAULT_MODEL: LazyLock<ModelId> = LazyLock::new(|| ModelId::new("gpt-4o-mini"));

/// Base URL for the OpenAI REST API.
pub const API_BASE: &str = "https://api.openai.com/v1";

/// The OAuth client-id registered for this application with OpenAI.
pub const OPENAI_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// Build the canonical [`ProviderDescriptor`] for OpenAI.
pub fn provider_descriptor() -> ProviderDescriptor {
    ProviderDescriptor {
        id: PROVIDER_ID.clone(),
        display_name: "OpenAI".to_owned(),
        default_model: DEFAULT_MODEL.clone(),
        capabilities: vec![
            ProviderCapability::ApiKeyAuth,
            ProviderCapability::OAuth,
            ProviderCapability::ToolCalling,
            ProviderCapability::Streaming,
            ProviderCapability::ModelListing,
            ProviderCapability::SystemPrompt,
        ],
        metadata: Metadata::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_id_is_openai() {
        assert_eq!(PROVIDER_ID.as_str(), "openai");
    }

    #[test]
    fn default_model_is_gpt4o_mini() {
        assert_eq!(DEFAULT_MODEL.as_str(), "gpt-4o-mini");
    }

    #[test]
    fn descriptor_has_expected_capabilities() {
        let desc = provider_descriptor();
        assert_eq!(desc.id.as_str(), "openai");
        assert!(desc.capabilities.contains(&ProviderCapability::ToolCalling));
        assert!(desc.capabilities.contains(&ProviderCapability::OAuth));
        assert!(desc.capabilities.contains(&ProviderCapability::ApiKeyAuth));
    }
}
