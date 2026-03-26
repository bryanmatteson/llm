use serde::{Deserialize, Serialize};

fn default_max_turns() -> usize {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDefaults {
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub tool_confirmation_required: bool,
}

impl Default for SessionDefaults {
    fn default() -> Self {
        Self {
            max_turns: default_max_turns(),
            system_prompt: None,
            tool_confirmation_required: false,
        }
    }
}
