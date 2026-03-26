use std::sync::LazyLock;

use llm_core::{
    Metadata, ModelId, ProviderCapability, ProviderDescriptor, ProviderId,
};

/// Provider identifier used across the framework.
pub static PROVIDER_ID: LazyLock<ProviderId> = LazyLock::new(|| ProviderId::new("google"));

/// The default model to use when none is specified.
pub static DEFAULT_MODEL: LazyLock<ModelId> = LazyLock::new(|| ModelId::new("gemini-2.5-flash"));

/// Base URL for the Google Gemini REST API.
pub const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta";

/// OAuth client ID for Google (installed application, same as Gemini CLI).
///
/// This is a public "installed application" client ID and is safe to embed
/// in source code.
pub const GOOGLE_CLIENT_ID: &str =
    "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com";

/// OAuth client secret for Google (installed application, same as Gemini CLI).
///
/// This is an "installed application" secret and is safe to embed in source.
/// See: <https://developers.google.com/identity/protocols/oauth2#installed>
pub const GOOGLE_CLIENT_SECRET: &str = "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";

/// Build the canonical [`ProviderDescriptor`] for Google Gemini.
pub fn provider_descriptor() -> ProviderDescriptor {
    ProviderDescriptor {
        id: PROVIDER_ID.clone(),
        display_name: "Google Gemini".to_owned(),
        default_model: DEFAULT_MODEL.clone(),
        capabilities: vec![
            ProviderCapability::ApiKeyAuth,
            ProviderCapability::OAuth,
            ProviderCapability::ToolCalling,
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
    fn provider_id_is_google() {
        assert_eq!(PROVIDER_ID.as_str(), "google");
    }

    #[test]
    fn default_model_is_gemini_flash() {
        assert_eq!(DEFAULT_MODEL.as_str(), "gemini-2.5-flash");
    }

    #[test]
    fn descriptor_has_expected_capabilities() {
        let desc = provider_descriptor();
        assert_eq!(desc.id.as_str(), "google");
        assert_eq!(desc.display_name, "Google Gemini");
        assert!(desc.capabilities.contains(&ProviderCapability::ToolCalling));
        assert!(desc.capabilities.contains(&ProviderCapability::OAuth));
        assert!(desc.capabilities.contains(&ProviderCapability::ApiKeyAuth));
        assert!(desc.capabilities.contains(&ProviderCapability::ModelListing));
        assert!(desc.capabilities.contains(&ProviderCapability::SystemPrompt));
    }
}
