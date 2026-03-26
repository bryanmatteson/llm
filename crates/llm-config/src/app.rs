use std::path::PathBuf;

use kdl_config::KdlNode;
use llm_core::ProviderId;
use serde::{Deserialize, Serialize};

use crate::provider::ProviderConfig;
use crate::session::SessionDefaults;
use crate::tool::ToolPolicyConfig;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, KdlNode)]
#[kdl(
    node = "llm",
    rename_all = "kebab-case",
    deny_unknown,
    skip_serialize_none,
    skip_serialize_empty_collections
)]
pub struct LlmConfig {
    #[kdl(attr, optional, from = "String")]
    pub default_provider: Option<ProviderId>,
    #[kdl(attr, optional)]
    pub data_dir: Option<PathBuf>,
    #[kdl(children, name = "provider", default)]
    pub providers: Vec<ProviderConfig>,
    #[kdl(child, default)]
    pub session_defaults: SessionDefaults,
    #[kdl(children, name = "tool-policy", default)]
    pub tool_policies: Vec<ToolPolicyConfig>,
}
