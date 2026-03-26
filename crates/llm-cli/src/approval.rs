use std::io::{self, BufRead, Write};

use async_trait::async_trait;

use llm_core::Result;
use llm_session::{ApprovalHandler, ApprovalRequest, ApprovalResponse};

/// CLI-based approval handler that prompts the user on stdin/stderr.
pub struct CliApprovalHandler;

#[async_trait]
impl ApprovalHandler for CliApprovalHandler {
    async fn request_approval(&self, request: ApprovalRequest) -> Result<ApprovalResponse> {
        // Print details to stderr so they don't mix with piped stdout.
        let stderr = io::stderr();
        let mut err = stderr.lock();

        writeln!(err).ok();
        writeln!(err, "Tool call requested:").ok();
        writeln!(err, "  tool:      {}", request.tool_name).ok();
        writeln!(err, "  call_id:   {}", request.call_id).ok();

        // Pretty-print arguments, falling back to compact form.
        let args_display = serde_json::to_string_pretty(&request.arguments)
            .unwrap_or_else(|_| request.arguments.to_string());
        for line in args_display.lines() {
            writeln!(err, "  {line}").ok();
        }

        write!(err, "Allow? [Y/n] ").ok();
        err.flush().ok();

        let stdin = io::stdin();
        let mut input = String::new();
        stdin.lock().read_line(&mut input).map_err(|e| {
            llm_core::FrameworkError::session(format!("failed to read approval input: {e}"))
        })?;

        let trimmed = input.trim().to_lowercase();

        if trimmed.is_empty() || trimmed == "y" || trimmed == "yes" {
            Ok(ApprovalResponse::Approve)
        } else {
            Ok(ApprovalResponse::Deny {
                reason: Some("user denied the tool call".into()),
            })
        }
    }
}
