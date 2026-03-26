use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use llm_core::{ModelId, ProviderId};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub id: ProviderId,
    pub display_name: String,
    pub auth_mode: AuthModeConfig,
    pub default_model: Option<ModelId>,
    pub base_url: Option<String>,
    pub api_key_env_var: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthModeConfig {
    ApiKey,
    OAuth,
    SessionToken,
}
