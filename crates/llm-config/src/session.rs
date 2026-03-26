use kdl_config::KdlNode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, KdlNode)]
#[kdl(node = "session-defaults", rename_all = "kebab-case", deny_unknown)]
pub struct SessionDefaults {
    #[kdl(attr, default = 10)]
    pub max_turns: i64,
    #[kdl(value, optional)]
    pub system_prompt: Option<String>,
    #[kdl(attr, flag, default)]
    pub tool_confirmation_required: bool,
}

impl Default for SessionDefaults {
    fn default() -> Self {
        Self {
            max_turns: 10,
            system_prompt: None,
            tool_confirmation_required: false,
        }
    }
}
