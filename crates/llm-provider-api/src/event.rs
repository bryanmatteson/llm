use serde::{Deserialize, Serialize};

use llm_core::{ModelId, StopReason, TokenUsage};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProviderEvent {
    TextDelta {
        text: String,
    },
    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallDelta {
        id: String,
        arguments_delta: String,
    },
    ToolCallEnd {
        id: String,
    },
    UsageReported(TokenUsage),
    Done {
        stop_reason: StopReason,
        model: ModelId,
    },
}
