use serde::{Deserialize, Serialize};
use serde_json::Value;

use llm_core::ToolId;

/// A tool call as requested by the model.
///
/// This mirrors the shape that provider APIs use: a call has an opaque `id`
/// assigned by the provider, the tool `name` (wire-format name), and the
/// JSON-encoded `arguments`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned identifier for this call (used to correlate the
    /// result back to the model).
    pub id: String,
    /// Wire-format name of the tool (matches [`ToolDescriptor::wire_name`]).
    pub name: String,
    /// JSON object containing the arguments the model supplied.
    pub arguments: Value,
}

/// The result of executing a tool, ready to be sent back to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The provider-assigned call id this result answers.
    pub call_id: String,
    /// The framework-level tool id that was executed.
    pub tool_id: ToolId,
    /// Primary textual content of the result (what the model will read).
    pub content: String,
    /// A short human-readable summary of the result (useful for logging or UI).
    pub summary: String,
    /// Arbitrary structured metadata about the execution (timings, token
    /// counts, etc.).
    pub metadata: Value,
}
