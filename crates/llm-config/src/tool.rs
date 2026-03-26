use serde::{Deserialize, Serialize};

use llm_core::ToolId;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPolicyConfig {
    pub tool_id: ToolId,
    pub allowed: bool,
    pub require_confirmation: bool,
    pub max_calls_per_session: Option<usize>,
}
