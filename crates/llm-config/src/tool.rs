use kdl_config::KdlNode;
use llm_core::ToolId;
use serde::{Deserialize, Serialize};

fn validate_access(cfg: &ToolPolicyConfig) -> Result<(), String> {
    if cfg.allowed.is_none() {
        Err(format!(
            "tool-policy '{}' must specify 'allow' or 'forbid'",
            cfg.tool_id
        ))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, KdlNode)]
#[kdl(
    node = "tool-policy",
    rename_all = "kebab-case",
    deny_unknown,
    validate(func = "validate_access")
)]
pub struct ToolPolicyConfig {
    #[kdl(attr, positional = 0, from = "String")]
    pub tool_id: ToolId,
    /// Explicit access: `allow` or `forbid`. Omitting both is a parse error.
    #[kdl(attr, flag = "allow", neg_flag = "forbid", optional)]
    pub allowed: Option<bool>,
    #[kdl(attr, flag, default)]
    pub require_confirmation: bool,
    #[kdl(attr, optional)]
    pub max_calls: Option<i64>,
}
