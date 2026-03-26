use std::io::{self, Write};

use llm_session::{EventReceiver, SessionEvent};

/// Read session events from the channel and render them to stderr.
///
/// This function runs until the sender is dropped (i.e. the channel is
/// closed).
pub async fn render_session_events(rx: &mut EventReceiver) {
    let stderr = io::stderr();
    let mut received_deltas = false;

    while let Some(event) = rx.recv().await {
        let mut err = stderr.lock();

        match event {
            SessionEvent::AssistantDelta { text } => {
                received_deltas = true;
                // Print streaming text inline without a newline so deltas
                // appear as a continuous stream.
                write!(err, "{text}").ok();
                err.flush().ok();
            }

            SessionEvent::ToolCallRequested {
                call_id: _,
                tool_name,
                arguments: _,
            } => {
                writeln!(err).ok();
                writeln!(err, "[calling tool: {tool_name}]").ok();
            }

            SessionEvent::ToolCallCompleted {
                call_id: _,
                tool_name: _,
                summary,
            } => {
                writeln!(err, "[tool result: {summary}]").ok();
            }

            SessionEvent::ToolApprovalRequired {
                call_id: _,
                tool_name,
                arguments: _,
            } => {
                writeln!(err, "[approval required for tool: {tool_name}]").ok();
            }

            SessionEvent::TurnCompleted { text, model, usage } => {
                // If streaming deltas were received, `text` duplicates what
                // was already printed.  Only print it when no deltas arrived.
                if !received_deltas {
                    writeln!(err).ok();
                    writeln!(err, "{text}").ok();
                }
                writeln!(err).ok();
                writeln!(
                    err,
                    "  [model={model} tokens_in={} tokens_out={}]",
                    usage.input_tokens, usage.output_tokens,
                )
                .ok();
            }

            SessionEvent::TurnLimitReached { turns_used } => {
                writeln!(err, "[turn limit reached after {turns_used} turns]").ok();
            }

            SessionEvent::Error { message } => {
                writeln!(err, "[error: {message}]").ok();
            }
        }
    }
}
