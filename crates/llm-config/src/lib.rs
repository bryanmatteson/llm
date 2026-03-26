pub mod app;
pub mod loader;
pub mod provider;
pub mod session;
pub mod tool;

pub use app::LlmConfig;
pub use loader::ConfigLoader;
pub use provider::{AuthMode, ProviderConfig};
pub use session::SessionDefaults;
pub use tool::ToolPolicyConfig;

#[cfg(test)]
mod tests {
    use crate::{ConfigLoader, LlmConfig};

    /// Canonical KDL surface for the LLM framework.
    ///
    /// - Identity is positional (`provider "openai"`, `tool-policy "web_search"`)
    /// - `auth-mode` is an inline discriminator (what kind of provider)
    /// - Provider settings are value children (display-name, base-url, etc.)
    /// - `system-prompt` is a value child (can be multi-paragraph)
    /// - `allow` / `forbid` are explicit, required flags on tool-policy
    /// - `deny_unknown` on all structs
    const SAMPLE_KDL: &str = r#"
llm default-provider="openai" {
    provider "openai" auth-mode="api-key" {
        display-name "OpenAI"
        default-model "gpt-4o"
        base-url "https://api.openai.com/v1"
        api-key-env-var "OPENAI_API_KEY"
    }

    provider "anthropic" auth-mode="api-key" {
        display-name "Anthropic"
        default-model "claude-sonnet-4-20250514"
        api-key-env-var "ANTHROPIC_API_KEY"
    }

    provider "google" auth-mode="oauth" {
        display-name "Google"
        default-model "gemini-2.5-flash"
    }

    session-defaults max-turns=20 tool-confirmation-required {
        system-prompt "You are a helpful assistant."
    }

    tool-policy "web_search" allow max-calls=5
    tool-policy "file_write" allow require-confirmation
    tool-policy "dangerous_exec" forbid
}
"#;

    #[test]
    fn parse_kdl_config() {
        let config: LlmConfig = ConfigLoader::parse(SAMPLE_KDL).expect("failed to parse KDL");

        assert_eq!(
            config.default_provider.as_ref().map(|p| p.as_str()),
            Some("openai")
        );

        // Providers
        assert_eq!(config.providers.len(), 3);
        assert_eq!(config.providers[0].id.as_str(), "openai");
        assert_eq!(config.providers[0].display_name, "OpenAI");
        assert_eq!(
            config.providers[0].auth_mode,
            crate::provider::AuthMode::ApiKey
        );
        assert_eq!(
            config.providers[0]
                .default_model
                .as_ref()
                .map(|m| m.as_str()),
            Some("gpt-4o")
        );
        assert_eq!(
            config.providers[0].base_url.as_deref(),
            Some("https://api.openai.com/v1")
        );

        assert_eq!(config.providers[1].id.as_str(), "anthropic");
        assert_eq!(
            config.providers[2].auth_mode,
            crate::provider::AuthMode::OAuth
        );

        // Session defaults
        assert_eq!(config.session_defaults.max_turns, 20);
        assert_eq!(
            config.session_defaults.system_prompt.as_deref(),
            Some("You are a helpful assistant.")
        );
        assert!(config.session_defaults.tool_confirmation_required);

        // Tool policies — explicit allow/forbid
        assert_eq!(config.tool_policies.len(), 3);

        assert_eq!(config.tool_policies[0].tool_id.as_str(), "web_search");
        assert_eq!(config.tool_policies[0].allowed, Some(true));
        assert!(!config.tool_policies[0].require_confirmation);
        assert_eq!(config.tool_policies[0].max_calls, Some(5));

        assert_eq!(config.tool_policies[1].tool_id.as_str(), "file_write");
        assert_eq!(config.tool_policies[1].allowed, Some(true));
        assert!(config.tool_policies[1].require_confirmation);

        assert_eq!(config.tool_policies[2].tool_id.as_str(), "dangerous_exec");
        assert_eq!(config.tool_policies[2].allowed, Some(false));
    }

    #[test]
    fn omitting_allow_forbid_is_a_parse_error() {
        let kdl = r#"
llm {
    tool-policy "ambiguous"
}
"#;
        let result = ConfigLoader::parse(kdl);
        assert!(
            result.is_err(),
            "should reject tool-policy without allow/forbid"
        );
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("allow") || err.contains("forbid"),
            "error should mention allow/forbid, got: {err}"
        );
    }

    #[test]
    fn defaults_when_minimal() {
        let config: LlmConfig =
            ConfigLoader::parse("llm {\n}").expect("failed to parse minimal KDL");

        assert!(config.default_provider.is_none());
        assert!(config.providers.is_empty());
        assert_eq!(config.session_defaults.max_turns, 10);
        assert!(!config.session_defaults.tool_confirmation_required);
        assert!(config.tool_policies.is_empty());
    }
}
