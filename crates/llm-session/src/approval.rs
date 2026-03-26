use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use llm_core::Result;

/// A request for human approval of a tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Provider-assigned call identifier.
    pub call_id: String,
    /// Wire-format name of the tool.
    pub tool_name: String,
    /// The arguments the model supplied for this call.
    pub arguments: serde_json::Value,
}

/// The response to an [`ApprovalRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalResponse {
    /// The tool call is approved and should proceed.
    Approve,
    /// The tool call is denied, optionally with a reason that will be relayed
    /// back to the model.
    Deny { reason: Option<String> },
}

/// Async trait for obtaining human approval for tool calls that require
/// confirmation.
///
/// This is strictly a *user interaction* boundary -- it is **not** the
/// policy engine.  The [`ToolPolicy`](llm_tools::ToolPolicy) decides
/// *whether* approval is needed; an `ApprovalHandler` decides *how* to
/// obtain it (e.g. CLI prompt, GUI dialog, auto-approve in tests).
#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    /// Ask the user whether the given tool call should proceed.
    async fn request_approval(&self, request: ApprovalRequest) -> Result<ApprovalResponse>;
}

/// An [`ApprovalHandler`] that unconditionally approves every request.
///
/// Useful for tests and non-interactive environments.
#[derive(Debug, Clone, Copy)]
pub struct AutoApproveHandler;

#[async_trait]
impl ApprovalHandler for AutoApproveHandler {
    async fn request_approval(&self, _request: ApprovalRequest) -> Result<ApprovalResponse> {
        Ok(ApprovalResponse::Approve)
    }
}
