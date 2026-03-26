use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use llm_core::ProviderId;

use crate::provider::ProviderConfig;
use crate::session::SessionDefaults;
use crate::tool::ToolPolicyConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub default_provider: Option<ProviderId>,
    pub config_dir: PathBuf,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub providers: Vec<ProviderConfig>,
    #[serde(default)]
    pub session_defaults: SessionDefaults,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_policies: Vec<ToolPolicyConfig>,
}
