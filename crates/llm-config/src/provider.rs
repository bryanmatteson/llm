use kdl_config::{Kdl, KdlNode, KdlValue};
use llm_core::{ModelId, ProviderId};
use serde::{Deserialize, Serialize};

/// How the provider authenticates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, KdlValue)]
#[kdl(rename_all = "kebab-case")]
pub enum AuthMode {
    ApiKey,
    OAuth,
    SessionToken,
}

/// Auth configuration child node.
///
/// ```kdl
/// auth mode=oauth session-file="openai-session.json"
/// ```
///
/// When `mode=api-key`, the key is resolved from the provider's standard
/// environment variable (e.g. `OPENAI_API_KEY`) or from the auth-dir.
/// `session-file` is optional and defaults to `<provider>-<name>.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, KdlNode)]
#[kdl(node = "auth", rename_all = "kebab-case", deny_unknown)]
pub struct AuthConfig {
    #[kdl(attr, scalar)]
    pub mode: AuthMode,
    #[kdl(attr, optional)]
    pub session_file: Option<String>,
    #[kdl(attr, optional)]
    pub env_var: Option<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            mode: AuthMode::ApiKey,
            session_file: None,
            env_var: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Kdl)]
#[kdl(node = "provider", rename_all = "kebab-case", deny_unknown)]
pub struct ProviderConfig {
    /// Identity — positional (e.g. `provider openai`).
    #[kdl(positional = 0, from = "String")]
    pub id: ProviderId,
    /// Optional session name (e.g. `name="session-gpt"`).
    /// Used to derive default session-file: `<provider>-<name>.json`.
    #[kdl(attr, optional)]
    pub name: Option<String>,
    /// Auth configuration.
    #[kdl(child, default)]
    pub auth: AuthConfig,
    /// Default model for this provider.
    #[kdl(value, optional, from = "String")]
    pub model: Option<ModelId>,
    /// Custom base URL for the provider API.
    #[kdl(value, optional)]
    pub base_url: Option<String>,
}
