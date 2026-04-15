use std::path::PathBuf;

use kdl_config::Kdl;
use llm_core::ProviderId;
use serde::{Deserialize, Serialize};

use crate::provider::ProviderConfig;
use crate::session::SessionDefaults;
use crate::tool::ToolPolicyConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Kdl)]
#[kdl(
    node = "llm",
    schema,
    rename_all = "kebab-case",
    deny_unknown,
    skip_serialize_none,
    skip_serialize_empty_collections
)]
pub struct LlmConfig {
    #[kdl(attr, optional, from = "String")]
    pub default_provider: Option<ProviderId>,
    #[kdl(value, optional)]
    pub auth_dir: Option<PathBuf>,
    #[kdl(children, name = "provider", default)]
    pub providers: Vec<ProviderConfig>,
    #[kdl(child, default)]
    pub session_defaults: SessionDefaults,
    #[kdl(children, name = "tool-policy", default)]
    pub tool_policies: Vec<ToolPolicyConfig>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            default_provider: None,
            auth_dir: None,
            providers: Vec::new(),
            session_defaults: SessionDefaults::default(),
            tool_policies: Vec::new(),
        }
    }
}
