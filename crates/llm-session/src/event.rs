use llm_core::{ModelId, TokenUsage};
use serde::{Deserialize, Serialize};

/// Events emitted by the turn loop so that callers (CLI, GUI, tests) can
/// observe progress without polling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SessionEvent {
    /// An incremental text chunk from the assistant.
    AssistantDelta {
        text: String,
    },

    /// The assistant has requested a tool call.
    ToolCallRequested {
        call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },

    /// A tool call finished executing.
    ToolCallCompleted {
        call_id: String,
        tool_name: String,
        summary: String,
    },

    /// A tool call requires human approval before it can proceed.
    ToolApprovalRequired {
        call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },

    /// A complete turn finished with a final assistant response.
    TurnCompleted {
        text: String,
        model: ModelId,
        usage: TokenUsage,
    },

    /// The turn loop has exhausted its maximum turn budget.
    TurnLimitReached {
        turns_used: usize,
    },

    /// An error occurred during the turn loop.
    Error {
        message: String,
    },
}

/// Channel sender for [`SessionEvent`]s.
pub type EventSender = tokio::sync::mpsc::UnboundedSender<SessionEvent>;

/// Channel receiver for [`SessionEvent`]s.
pub type EventReceiver = tokio::sync::mpsc::UnboundedReceiver<SessionEvent>;

/// Create a new unbounded event channel.
pub fn event_channel() -> (EventSender, EventReceiver) {
    tokio::sync::mpsc::unbounded_channel()
}
