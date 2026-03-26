pub mod app;
pub mod loader;
pub mod provider;
pub mod session;
pub mod tool;

pub use app::AppConfig;
pub use loader::ConfigLoader;
pub use provider::{AuthModeConfig, ProviderConfig};
pub use session::SessionDefaults;
pub use tool::ToolPolicyConfig;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use llm_core::{ModelId, ProviderId, ToolId};

    use crate::{
        AppConfig, AuthModeConfig, ConfigLoader, ProviderConfig, SessionDefaults, ToolPolicyConfig,
    };

    #[test]
    fn toml_roundtrip() {
        let config = AppConfig {
            default_provider: Some(ProviderId::new("openai")),
            config_dir: PathBuf::from("/tmp/llm-config-test"),
            providers: vec![ProviderConfig {
                id: ProviderId::new("openai"),
                display_name: "OpenAI".to_string(),
                auth_mode: AuthModeConfig::ApiKey,
                default_model: Some(ModelId::new("gpt-4o")),
                base_url: Some("https://api.openai.com/v1".to_string()),
                api_key_env_var: Some("OPENAI_API_KEY".to_string()),
                extra: Default::default(),
            }],
            session_defaults: SessionDefaults {
                max_turns: 20,
                system_prompt: Some("You are a helpful assistant.".to_string()),
                tool_confirmation_required: true,
            },
            tool_policies: vec![ToolPolicyConfig {
                tool_id: ToolId::new("web_search"),
                allowed: true,
                require_confirmation: false,
                max_calls_per_session: Some(5),
            }],
        };

        let toml_str =
            toml::to_string_pretty(&config).expect("failed to serialize config to TOML");
        let deserialized: AppConfig =
            toml::from_str(&toml_str).expect("failed to deserialize config from TOML");

        assert_eq!(
            config.default_provider.as_ref().unwrap().as_str(),
            deserialized.default_provider.as_ref().unwrap().as_str(),
        );
        assert_eq!(config.config_dir, deserialized.config_dir);
        assert_eq!(config.providers.len(), deserialized.providers.len());
        assert_eq!(
            config.providers[0].display_name,
            deserialized.providers[0].display_name,
        );
        assert_eq!(
            config.session_defaults.max_turns,
            deserialized.session_defaults.max_turns,
        );
        assert_eq!(
            config.session_defaults.system_prompt,
            deserialized.session_defaults.system_prompt,
        );
        assert_eq!(
            config.session_defaults.tool_confirmation_required,
            deserialized.session_defaults.tool_confirmation_required,
        );
        assert_eq!(config.tool_policies.len(), deserialized.tool_policies.len());
        assert_eq!(
            config.tool_policies[0].tool_id.as_str(),
            deserialized.tool_policies[0].tool_id.as_str(),
        );
        assert_eq!(
            config.tool_policies[0].max_calls_per_session,
            deserialized.tool_policies[0].max_calls_per_session,
        );
    }

    #[test]
    fn loader_save_and_load() {
        let dir = std::env::temp_dir().join("llm-config-loader-test");
        let path = dir.join("config.toml");

        let config = AppConfig {
            default_provider: None,
            config_dir: PathBuf::from("/tmp"),
            providers: vec![],
            session_defaults: SessionDefaults::default(),
            tool_policies: vec![],
        };

        ConfigLoader::save_to_file(&config, &path).expect("failed to save config");
        let loaded = ConfigLoader::load_from_file(&path).expect("failed to load config");

        assert_eq!(config.config_dir, loaded.config_dir);
        assert!(loaded.default_provider.is_none());
        assert_eq!(loaded.session_defaults.max_turns, 10);
        assert!(!loaded.session_defaults.tool_confirmation_required);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }
}
