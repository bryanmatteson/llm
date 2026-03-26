use kdl_config::{Kdl, KdlValue};
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Kdl)]
#[kdl(node = "provider", rename_all = "kebab-case", deny_unknown)]
pub struct ProviderConfig {
    /// Identity — positional.
    #[kdl(positional = 0, from = "String")]
    pub id: ProviderId,
    /// Discriminator — stays inline because it modifies what the provider IS.
    #[kdl(attr, scalar)]
    pub auth_mode: AuthMode,
    /// Settings — value children, not identity.
    #[kdl(value)]
    pub display_name: String,
    #[kdl(value, optional, from = "String")]
    pub default_model: Option<ModelId>,
    #[kdl(value, optional)]
    pub base_url: Option<String>,
    #[kdl(value, optional)]
    pub api_key_env_var: Option<String>,
}
