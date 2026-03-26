use std::collections::HashMap;

use tokio_stream::StreamExt;

use llm_core::{
    ContentBlock, FrameworkError, Message, ModelId, Result, Role, StopReason, TokenUsage,
};
use llm_provider_api::{
    LlmProviderClient, ProviderEvent, ProviderToolDescriptor, ToolSchemaAdapter, TurnRequest,
};
use llm_tools::{ToolApproval, ToolContext, ToolRegistry};

use crate::approval::{ApprovalHandler, ApprovalRequest, ApprovalResponse};
use crate::config::SessionConfig;
use crate::conversation::ConversationState;
use crate::event::{EventSender, SessionEvent};

/// All the shared state and dependencies needed to run a turn loop.
///
/// Collects what would otherwise be 8 positional arguments into a
/// single struct, making call sites self-documenting.
pub struct TurnLoopContext<'a> {
    pub session_id: &'a llm_core::SessionId,
    pub client: &'a dyn LlmProviderClient,
    pub conversation: &'a mut ConversationState,
    pub tool_registry: &'a ToolRegistry,
    pub tool_adapter: &'a dyn ToolSchemaAdapter,
    pub config: &'a SessionConfig,
    pub approval_handler: &'a dyn ApprovalHandler,
    pub event_tx: Option<&'a EventSender>,
}

/// The outcome of a complete turn-loop execution.
#[derive(Debug, Clone)]
pub struct TurnOutcome {
    /// The final assistant text once the loop terminates.
    pub final_text: String,
    /// The model that produced the final response.
    pub model: ModelId,
    /// Aggregated token usage across all turns.
    pub usage: TokenUsage,
    /// Total number of tool calls executed across all turns.
    pub tool_calls_made: usize,
    /// Number of LLM round-trips performed.
    pub turns_used: usize,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Emit a [`SessionEvent`] if a sender is available.
fn emit(tx: Option<&EventSender>, event: SessionEvent) {
    if let Some(tx) = tx {
        // Best-effort: if the receiver has been dropped we silently ignore.
        let _ = tx.send(event);
    }
}

/// Build a [`TurnRequest`] from the current conversation state and config.
fn build_turn_request(
    conversation: &ConversationState,
    config: &SessionConfig,
    tool_descriptors_json: &[serde_json::Value],
) -> TurnRequest {
    TurnRequest {
        system_prompt: config.system_prompt.clone(),
        messages: conversation.messages().to_vec(),
        tools: tool_descriptors_json.to_vec(),
        model: config.model.clone(),
        max_tokens: None,
        temperature: None,
    }
}

/// Extract tool-use blocks from assistant messages returned by the provider.
fn extract_tool_calls(messages: &[Message]) -> Vec<(String, String, serde_json::Value)> {
    let mut calls = Vec::new();
    for msg in messages {
        for block in &msg.content {
            if let ContentBlock::ToolUse { id, name, input } = block {
                calls.push((id.clone(), name.clone(), input.clone()));
            }
        }
    }
    calls
}

/// Translate the registry's tool descriptors into the provider-specific JSON
/// format via the [`ToolSchemaAdapter`].
fn translate_tools(
    registry: &ToolRegistry,
    adapter: &dyn ToolSchemaAdapter,
) -> Vec<serde_json::Value> {
    let descriptors: Vec<ProviderToolDescriptor> = registry
        .all_descriptors()
        .into_iter()
        .map(|d| ProviderToolDescriptor {
            name: d.wire_name,
            description: d.description,
            parameters: d.parameters,
        })
        .collect();
    adapter.translate_descriptors(&descriptors)
}

/// Result of executing a batch of tool calls within a single turn.
struct ToolExecutionResult {
    /// Number of tool calls executed (including skipped/denied).
    calls_made: usize,
}

/// Execute a batch of tool calls, appending results to the conversation.
///
/// This is the shared tool-execution logic used by both `run_turn_loop` and
/// `run_streaming_turn_loop`.
async fn execute_tool_calls(
    calls: &[(String, String, serde_json::Value)],
    ctx: &mut TurnLoopContext<'_>,
    last_model: &ModelId,
) -> Result<ToolExecutionResult> {
    let mut tool_calls_this_turn: usize = 0;
    let mut total_made: usize = 0;

    for (call_id, tool_name, arguments) in calls {
        if tool_calls_this_turn >= ctx.config.limits.max_tool_calls_per_turn {
            let err_msg = format!(
                "Tool call limit ({}) reached; call to '{}' was skipped.",
                ctx.config.limits.max_tool_calls_per_turn, tool_name
            );
            ctx.conversation.append_tool_result(call_id, &err_msg);
            total_made += 1;
            continue;
        }

        emit(
            ctx.event_tx,
            SessionEvent::ToolCallRequested {
                call_id: call_id.clone(),
                tool_name: tool_name.clone(),
                arguments: arguments.clone(),
            },
        );

        // Look up the tool by wire name.
        let tool = match ctx.tool_registry.get_by_wire_name(tool_name) {
            Some(t) => t,
            None => {
                let err_msg = format!("Tool '{tool_name}' is not available.");
                ctx.conversation.append_tool_result(call_id, &err_msg);
                emit(ctx.event_tx, SessionEvent::Error { message: err_msg });
                total_made += 1;
                tool_calls_this_turn += 1;
                continue;
            }
        };

        let descriptor = tool.descriptor();

        // -- Policy check --
        let approval = ctx.config.tool_policy.approval_for(&descriptor.id);

        match approval {
            ToolApproval::Deny => {
                let deny_msg = format!("Tool '{tool_name}' is denied by policy.");
                ctx.conversation.append_tool_result(call_id, &deny_msg);
                emit(ctx.event_tx, SessionEvent::Error { message: deny_msg });
                total_made += 1;
                tool_calls_this_turn += 1;
                continue;
            }
            ToolApproval::RequireConfirmation => {
                emit(
                    ctx.event_tx,
                    SessionEvent::ToolApprovalRequired {
                        call_id: call_id.clone(),
                        tool_name: tool_name.clone(),
                        arguments: arguments.clone(),
                    },
                );

                let approval_resp = ctx
                    .approval_handler
                    .request_approval(ApprovalRequest {
                        call_id: call_id.clone(),
                        tool_name: tool_name.clone(),
                        arguments: arguments.clone(),
                    })
                    .await?;

                match approval_resp {
                    ApprovalResponse::Approve => {
                        // Fall through to execution below.
                    }
                    ApprovalResponse::Deny { reason } => {
                        let deny_msg = reason
                            .unwrap_or_else(|| format!("User denied tool call '{tool_name}'."));
                        ctx.conversation.append_tool_result(call_id, &deny_msg);
                        total_made += 1;
                        tool_calls_this_turn += 1;
                        continue;
                    }
                }
            }
            ToolApproval::Auto => {
                // Proceed directly.
            }
        }

        // -- Execute the tool --
        let tool_context = ToolContext {
            session_id: ctx.session_id.clone(),
            provider_id: ctx.config.provider_id.clone(),
            model_id: last_model.clone(),
            metadata: Default::default(),
        };

        let exec_result = match tokio::time::timeout(
            ctx.config.limits.tool_timeout,
            tool.invoke(arguments.clone(), &tool_context),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(FrameworkError::tool(
                descriptor.id.clone(),
                format!(
                    "execution timed out after {:?}",
                    ctx.config.limits.tool_timeout
                ),
            )),
        };

        let (content, summary) = match exec_result {
            Ok(value) => {
                let content_str =
                    serde_json::to_string(&value).unwrap_or_else(|_| value.to_string());
                let summary = content_str.chars().take(120).collect::<String>();
                (content_str, summary)
            }
            Err(e) => {
                let err_str = format!("Tool execution error: {e}");
                (err_str.clone(), err_str)
            }
        };

        ctx.conversation.append_tool_result(call_id, &content);

        emit(
            ctx.event_tx,
            SessionEvent::ToolCallCompleted {
                call_id: call_id.clone(),
                tool_name: tool_name.clone(),
                summary,
            },
        );

        total_made += 1;
        tool_calls_this_turn += 1;
    }

    Ok(ToolExecutionResult {
        calls_made: total_made,
    })
}

// ---------------------------------------------------------------------------
// Turn loop (non-streaming)
// ---------------------------------------------------------------------------

/// Execute the multi-step provider/tool turn loop.
///
/// This is the central orchestration function of `llm-session`.  It sends
/// conversation turns to the provider, dispatches tool calls (subject to the
/// configured [`ToolPolicy`](llm_tools::ToolPolicy) and
/// [`ApprovalHandler`]), appends results to the conversation, and repeats
/// until the model produces a final text response or a limit is reached.
pub async fn run_turn_loop(mut ctx: TurnLoopContext<'_>) -> Result<TurnOutcome> {
    let tool_descriptors_json = translate_tools(ctx.tool_registry, ctx.tool_adapter);

    let mut total_tool_calls: usize = 0;
    let mut aggregated_usage = TokenUsage::default();
    // Initialized to a fallback; overwritten on each turn by the provider's
    // reported model.  The initial value is only used if the loop exits
    // without completing a single turn (which cannot happen given
    // max_turns >= 1), but we need a value to satisfy the compiler.
    let mut last_model = ctx
        .config
        .model
        .clone()
        .unwrap_or_else(|| ModelId::new("unknown"));
    let _ = &last_model; // suppress unused-assignment warning

    for turn_index in 0..ctx.config.limits.max_turns {
        // 1. Build the request.
        let request = build_turn_request(ctx.conversation, ctx.config, &tool_descriptors_json);

        // 2. Call the provider.
        let response = tokio::time::timeout(
            ctx.config.limits.turn_timeout,
            ctx.client.send_turn(&request),
        )
        .await
        .map_err(|_| {
            FrameworkError::session(format!(
                "turn timed out after {:?}",
                ctx.config.limits.turn_timeout
            ))
        })??;

        // Accumulate usage.
        aggregated_usage.input_tokens += response.usage.input_tokens;
        aggregated_usage.output_tokens += response.usage.output_tokens;
        last_model = response.model.clone();

        // 3. Normalize: append the provider's response messages to the
        //    conversation transcript.
        for msg in &response.messages {
            ctx.conversation.append_message(msg.clone());
        }

        // 4. Branch on stop reason.
        match response.stop_reason {
            StopReason::ToolUse => {
                let calls = extract_tool_calls(&response.messages);

                if calls.is_empty() {
                    // Provider said ToolUse but gave no tool calls -- treat
                    // as end of turn to avoid infinite loops.
                    let final_text = response
                        .messages
                        .last()
                        .map(|m| m.text_content())
                        .unwrap_or_default();
                    emit(
                        ctx.event_tx,
                        SessionEvent::TurnCompleted {
                            text: final_text.clone(),
                            model: last_model.clone(),
                            usage: aggregated_usage.clone(),
                        },
                    );
                    return Ok(TurnOutcome {
                        final_text,
                        model: last_model,
                        usage: aggregated_usage,
                        tool_calls_made: total_tool_calls,
                        turns_used: turn_index + 1,
                    });
                }

                let result = execute_tool_calls(&calls, &mut ctx, &last_model).await?;
                total_tool_calls += result.calls_made;

                // Continue the loop -- the provider needs to see the tool
                // results and generate a follow-up.
                continue;
            }

            StopReason::EndTurn | StopReason::Stop | StopReason::MaxTokens => {
                let final_text = response
                    .messages
                    .last()
                    .map(|m| m.text_content())
                    .unwrap_or_default();

                emit(
                    ctx.event_tx,
                    SessionEvent::TurnCompleted {
                        text: final_text.clone(),
                        model: last_model.clone(),
                        usage: aggregated_usage.clone(),
                    },
                );

                return Ok(TurnOutcome {
                    final_text,
                    model: last_model,
                    usage: aggregated_usage,
                    tool_calls_made: total_tool_calls,
                    turns_used: turn_index + 1,
                });
            }
        }
    }

    // Exhausted the turn budget.
    emit(
        ctx.event_tx,
        SessionEvent::TurnLimitReached {
            turns_used: ctx.config.limits.max_turns,
        },
    );

    Err(FrameworkError::session(format!(
        "turn limit reached after {} turns",
        ctx.config.limits.max_turns
    )))
}

// ---------------------------------------------------------------------------
// Streaming turn loop
// ---------------------------------------------------------------------------

/// Accumulated state while consuming a stream of [`ProviderEvent`]s for a
/// single turn.
struct StreamAccumulator {
    /// Incremental text content from `TextDelta` events.
    text: String,
    /// Completed tool calls: `(id, name, parsed_args_json)`.
    tool_calls: Vec<(String, String, serde_json::Value)>,
    /// Per-tool-call argument fragments keyed by call ID.
    current_tool_args: HashMap<String, String>,
    /// Per-tool-call names keyed by call ID (from `ToolCallStart`).
    current_tool_names: HashMap<String, String>,
    /// Accumulated token usage for this turn.
    usage: TokenUsage,
    /// Stop reason reported by the `Done` event.
    stop_reason: Option<StopReason>,
    /// Model ID reported by the `Done` event.
    model: Option<ModelId>,
}

impl StreamAccumulator {
    fn new() -> Self {
        Self {
            text: String::new(),
            tool_calls: Vec::new(),
            current_tool_args: HashMap::new(),
            current_tool_names: HashMap::new(),
            usage: TokenUsage::default(),
            stop_reason: None,
            model: None,
        }
    }

    /// Build an assistant [`Message`] from the accumulated text and tool calls.
    fn into_message(self) -> Message {
        let mut blocks = Vec::new();

        if !self.text.is_empty() {
            blocks.push(ContentBlock::Text(self.text));
        }

        for (id, name, input) in &self.tool_calls {
            blocks.push(ContentBlock::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            });
        }

        Message {
            role: Role::Assistant,
            content: blocks,
            metadata: Default::default(),
        }
    }
}

/// Execute the multi-step provider/tool turn loop using **streaming**.
///
/// This is the streaming counterpart to [`run_turn_loop`].  Instead of calling
/// `send_turn` for a complete response, it calls `stream_turn` and processes
/// incremental [`ProviderEvent`]s, emitting [`SessionEvent::AssistantDelta`]
/// events for each text chunk so that UIs can display partial output.
///
/// If the provider does not support streaming (returns
/// [`FrameworkError::Unsupported`]), the function transparently falls back to
/// `send_turn` for that turn.
///
/// All tool execution, policy checking, and approval flow is shared with the
/// non-streaming path via [`execute_tool_calls`].
pub async fn run_streaming_turn_loop(mut ctx: TurnLoopContext<'_>) -> Result<TurnOutcome> {
    let tool_descriptors_json = translate_tools(ctx.tool_registry, ctx.tool_adapter);

    let mut total_tool_calls: usize = 0;
    let mut aggregated_usage = TokenUsage::default();
    let mut last_model = ctx
        .config
        .model
        .clone()
        .unwrap_or_else(|| ModelId::new("unknown"));
    let _ = &last_model;

    for turn_index in 0..ctx.config.limits.max_turns {
        let request = build_turn_request(ctx.conversation, ctx.config, &tool_descriptors_json);

        // A single timeout wraps both stream creation AND consumption so
        // the total wall-clock budget for one turn is exactly `turn_timeout`.
        let stream_result = tokio::time::timeout(
            ctx.config.limits.turn_timeout,
            async {
                let stream = ctx.client.stream_turn(&request).await?;
                consume_stream(stream, ctx.event_tx).await
            },
        )
        .await;

        let (turn_text, turn_tool_calls, turn_usage, turn_stop_reason, turn_model) =
            match stream_result {
                Ok(Ok(acc)) => {

                    let text = acc.text.clone();
                    let tool_calls = acc.tool_calls.clone();
                    let usage = acc.usage.clone();
                    let stop_reason = acc.stop_reason.unwrap_or(StopReason::EndTurn);
                    let model = acc.model.clone();

                    // Append the accumulated assistant message to the conversation.
                    let msg = acc.into_message();
                    ctx.conversation.append_message(msg);

                    (text, tool_calls, usage, stop_reason, model)
                }
                Ok(Err(FrameworkError::Unsupported { .. })) => {
                    // Graceful degradation: fall back to non-streaming.
                    let response = tokio::time::timeout(
                        ctx.config.limits.turn_timeout,
                        ctx.client.send_turn(&request),
                    )
                    .await
                    .map_err(|_| {
                        FrameworkError::session(format!(
                            "turn timed out after {:?}",
                            ctx.config.limits.turn_timeout
                        ))
                    })??;

                    for msg in &response.messages {
                        ctx.conversation.append_message(msg.clone());
                    }

                    let text = response
                        .messages
                        .last()
                        .map(|m| m.text_content())
                        .unwrap_or_default();
                    let tool_calls = extract_tool_calls(&response.messages);
                    (
                        text,
                        tool_calls,
                        response.usage,
                        response.stop_reason,
                        Some(response.model),
                    )
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(FrameworkError::session(format!(
                        "streaming turn timed out after {:?}",
                        ctx.config.limits.turn_timeout
                    )));
                }
            };

        // Accumulate usage.
        aggregated_usage.input_tokens += turn_usage.input_tokens;
        aggregated_usage.output_tokens += turn_usage.output_tokens;
        if let Some(ref m) = turn_model {
            last_model = m.clone();
        }

        // Branch on stop reason.
        match turn_stop_reason {
            StopReason::ToolUse => {
                if turn_tool_calls.is_empty() {
                    emit(
                        ctx.event_tx,
                        SessionEvent::TurnCompleted {
                            text: turn_text.clone(),
                            model: last_model.clone(),
                            usage: aggregated_usage.clone(),
                        },
                    );
                    return Ok(TurnOutcome {
                        final_text: turn_text,
                        model: last_model,
                        usage: aggregated_usage,
                        tool_calls_made: total_tool_calls,
                        turns_used: turn_index + 1,
                    });
                }

                let result = execute_tool_calls(&turn_tool_calls, &mut ctx, &last_model).await?;
                total_tool_calls += result.calls_made;
                continue;
            }
            StopReason::EndTurn | StopReason::Stop | StopReason::MaxTokens => {
                emit(
                    ctx.event_tx,
                    SessionEvent::TurnCompleted {
                        text: turn_text.clone(),
                        model: last_model.clone(),
                        usage: aggregated_usage.clone(),
                    },
                );

                return Ok(TurnOutcome {
                    final_text: turn_text,
                    model: last_model,
                    usage: aggregated_usage,
                    tool_calls_made: total_tool_calls,
                    turns_used: turn_index + 1,
                });
            }
        }
    }

    emit(
        ctx.event_tx,
        SessionEvent::TurnLimitReached {
            turns_used: ctx.config.limits.max_turns,
        },
    );

    Err(FrameworkError::session(format!(
        "turn limit reached after {} turns",
        ctx.config.limits.max_turns
    )))
}

/// Consume a stream of [`ProviderEvent`]s into a [`StreamAccumulator`],
/// emitting [`SessionEvent::AssistantDelta`] for each text chunk.
async fn consume_stream(
    mut stream: std::pin::Pin<Box<dyn tokio_stream::Stream<Item = Result<ProviderEvent>> + Send>>,
    event_tx: Option<&EventSender>,
) -> Result<StreamAccumulator> {
    let mut acc = StreamAccumulator::new();

    while let Some(event_result) = stream.next().await {
        let event = event_result?;
        match event {
            ProviderEvent::TextDelta { text } => {
                emit(
                    event_tx,
                    SessionEvent::AssistantDelta { text: text.clone() },
                );
                acc.text.push_str(&text);
            }
            ProviderEvent::ToolCallStart { id, name } => {
                acc.current_tool_names.insert(id.clone(), name);
                acc.current_tool_args.insert(id, String::new());
            }
            ProviderEvent::ToolCallDelta {
                id,
                arguments_delta,
            } => {
                acc.current_tool_args
                    .entry(id)
                    .or_default()
                    .push_str(&arguments_delta);
            }
            ProviderEvent::ToolCallEnd { id } => {
                let args_json = acc.current_tool_args.remove(&id).unwrap_or_default();
                let name = acc.current_tool_names.remove(&id).unwrap_or_default();
                let parsed: serde_json::Value = serde_json::from_str(&args_json)
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                acc.tool_calls.push((id, name, parsed));
            }
            ProviderEvent::UsageReported(usage) => {
                acc.usage.input_tokens += usage.input_tokens;
                acc.usage.output_tokens += usage.output_tokens;
            }
            ProviderEvent::Done { stop_reason, model } => {
                acc.stop_reason = Some(stop_reason);
                acc.model = Some(model);
            }
        }
    }

    Ok(acc)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::pin::Pin;

    use async_trait::async_trait;
    use tokio_stream::Stream;

    use llm_core::{ModelDescriptor, ModelId, ProviderId, StopReason, TokenUsage};
    use llm_provider_api::{
        LlmProviderClient, ProviderEvent, ProviderToolDescriptor, ToolSchemaAdapter, TurnRequest,
        TurnResponse,
    };
    use llm_tools::ToolRegistry;

    use super::*;
    use crate::approval::AutoApproveHandler;
    use crate::config::SessionConfig;
    use crate::conversation::ConversationState;
    use crate::limits::SessionLimits;

    // -- Mock provider client -----------------------------------------------

    /// A mock that always returns a simple text response with `EndTurn`.
    #[derive(Debug)]
    struct MockClient {
        provider_id: ProviderId,
        response_text: String,
        model: ModelId,
    }

    #[async_trait]
    impl LlmProviderClient for MockClient {
        fn provider_id(&self) -> &ProviderId {
            &self.provider_id
        }

        async fn send_turn(&self, _request: &TurnRequest) -> llm_core::Result<TurnResponse> {
            Ok(TurnResponse {
                messages: vec![llm_core::Message::assistant(&self.response_text)],
                stop_reason: StopReason::EndTurn,
                model: self.model.clone(),
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
            })
        }

        async fn stream_turn(
            &self,
            _request: &TurnRequest,
        ) -> llm_core::Result<Pin<Box<dyn Stream<Item = llm_core::Result<ProviderEvent>> + Send>>>
        {
            Err(FrameworkError::unsupported(
                "streaming not implemented in mock",
            ))
        }

        async fn list_models(&self) -> llm_core::Result<Vec<ModelDescriptor>> {
            Ok(vec![])
        }
    }

    // -- Mock tool schema adapter -------------------------------------------

    /// A passthrough adapter that wraps each descriptor as-is.
    #[derive(Debug)]
    struct PassthroughAdapter;

    impl ToolSchemaAdapter for PassthroughAdapter {
        fn translate_descriptors(
            &self,
            tools: &[ProviderToolDescriptor],
        ) -> Vec<serde_json::Value> {
            tools
                .iter()
                .map(|t| serde_json::to_value(t).unwrap())
                .collect()
        }

        fn parse_tool_calls(
            &self,
            _response: &serde_json::Value,
        ) -> Vec<llm_provider_api::ProviderToolCall> {
            vec![]
        }
    }

    // -- Tests --------------------------------------------------------------

    #[tokio::test]
    async fn simple_text_response_completes_in_one_turn() {
        let client = MockClient {
            provider_id: ProviderId::new("test"),
            response_text: "Hello from the mock!".to_string(),
            model: ModelId::new("mock-model"),
        };

        let mut conversation = ConversationState::new();
        conversation.append_user("Hi there");

        let registry = ToolRegistry::new();
        let adapter = PassthroughAdapter;
        let config = SessionConfig {
            provider_id: ProviderId::new("test"),
            model: Some(ModelId::new("mock-model")),
            system_prompt: None,
            tool_policy: Default::default(),
            limits: SessionLimits::default(),
            metadata: Default::default(),
        };
        let approval = AutoApproveHandler;

        let (tx, mut rx) = crate::event::event_channel();

        let session_id = llm_core::SessionId::new("test-session");
        let outcome = run_turn_loop(TurnLoopContext {
            session_id: &session_id,
            client: &client,
            conversation: &mut conversation,
            tool_registry: &registry,
            tool_adapter: &adapter,
            config: &config,
            approval_handler: &approval,
            event_tx: Some(&tx),
        })
        .await
        .expect("turn loop should succeed");

        assert_eq!(outcome.final_text, "Hello from the mock!");
        assert_eq!(outcome.model.as_str(), "mock-model");
        assert_eq!(outcome.turns_used, 1);
        assert_eq!(outcome.tool_calls_made, 0);
        assert_eq!(outcome.usage.input_tokens, 10);
        assert_eq!(outcome.usage.output_tokens, 5);

        // Verify a TurnCompleted event was emitted.
        drop(tx);
        let mut got_turn_completed = false;
        while let Some(event) = rx.recv().await {
            if matches!(event, SessionEvent::TurnCompleted { .. }) {
                got_turn_completed = true;
            }
        }
        assert!(got_turn_completed, "expected a TurnCompleted event");

        // Conversation should have user + assistant = 2 messages.
        assert_eq!(conversation.len(), 2);
    }

    // -- Streaming mock provider client ------------------------------------

    /// A mock that returns a stream of `ProviderEvent`s simulating a streamed
    /// text response.
    #[derive(Debug)]
    struct StreamingMockClient {
        provider_id: ProviderId,
        /// Individual text chunks to emit as `TextDelta`.
        text_chunks: Vec<String>,
        model: ModelId,
        usage: TokenUsage,
    }

    #[async_trait]
    impl LlmProviderClient for StreamingMockClient {
        fn provider_id(&self) -> &ProviderId {
            &self.provider_id
        }

        async fn send_turn(&self, _request: &TurnRequest) -> llm_core::Result<TurnResponse> {
            // Should not be called when streaming succeeds, but provide a
            // sensible fallback just in case.
            let full_text: String = self.text_chunks.concat();
            Ok(TurnResponse {
                messages: vec![llm_core::Message::assistant(&full_text)],
                stop_reason: StopReason::EndTurn,
                model: self.model.clone(),
                usage: self.usage.clone(),
            })
        }

        async fn stream_turn(
            &self,
            _request: &TurnRequest,
        ) -> llm_core::Result<Pin<Box<dyn Stream<Item = llm_core::Result<ProviderEvent>> + Send>>>
        {
            let mut events: Vec<llm_core::Result<ProviderEvent>> = Vec::new();

            for chunk in &self.text_chunks {
                events.push(Ok(ProviderEvent::TextDelta {
                    text: chunk.clone(),
                }));
            }

            events.push(Ok(ProviderEvent::UsageReported(self.usage.clone())));
            events.push(Ok(ProviderEvent::Done {
                stop_reason: StopReason::EndTurn,
                model: self.model.clone(),
            }));

            Ok(Box::pin(tokio_stream::iter(events)))
        }

        async fn list_models(&self) -> llm_core::Result<Vec<ModelDescriptor>> {
            Ok(vec![])
        }
    }

    // -- Streaming tests ---------------------------------------------------

    #[tokio::test]
    async fn streaming_text_response_emits_deltas_and_completes() {
        let client = StreamingMockClient {
            provider_id: ProviderId::new("test"),
            text_chunks: vec!["Hello".to_string(), " world".to_string()],
            model: ModelId::new("stream-model"),
            usage: TokenUsage {
                input_tokens: 12,
                output_tokens: 8,
            },
        };

        let mut conversation = ConversationState::new();
        conversation.append_user("Hi there");

        let registry = ToolRegistry::new();
        let adapter = PassthroughAdapter;
        let config = SessionConfig {
            provider_id: ProviderId::new("test"),
            model: Some(ModelId::new("stream-model")),
            system_prompt: None,
            tool_policy: Default::default(),
            limits: SessionLimits::default(),
            metadata: Default::default(),
        };
        let approval = AutoApproveHandler;

        let (tx, mut rx) = crate::event::event_channel();

        let session_id = llm_core::SessionId::new("test-streaming");
        let outcome = run_streaming_turn_loop(TurnLoopContext {
            session_id: &session_id,
            client: &client,
            conversation: &mut conversation,
            tool_registry: &registry,
            tool_adapter: &adapter,
            config: &config,
            approval_handler: &approval,
            event_tx: Some(&tx),
        })
        .await
        .expect("streaming turn loop should succeed");

        // Verify the concatenated text.
        assert_eq!(outcome.final_text, "Hello world");
        assert_eq!(outcome.model.as_str(), "stream-model");
        assert_eq!(outcome.turns_used, 1);
        assert_eq!(outcome.tool_calls_made, 0);

        // Verify usage was accumulated.
        assert_eq!(outcome.usage.input_tokens, 12);
        assert_eq!(outcome.usage.output_tokens, 8);

        // Verify events: AssistantDelta for each chunk, then TurnCompleted.
        drop(tx);
        let mut deltas = Vec::new();
        let mut got_turn_completed = false;
        while let Some(event) = rx.recv().await {
            match event {
                SessionEvent::AssistantDelta { text } => deltas.push(text),
                SessionEvent::TurnCompleted { .. } => got_turn_completed = true,
                _ => {}
            }
        }
        assert_eq!(deltas, vec!["Hello", " world"]);
        assert!(got_turn_completed, "expected a TurnCompleted event");

        // Conversation should have user + assistant = 2 messages.
        assert_eq!(conversation.len(), 2);
    }

    #[tokio::test]
    async fn streaming_falls_back_to_send_turn_when_unsupported() {
        // Use the original MockClient whose stream_turn returns Unsupported.
        let client = MockClient {
            provider_id: ProviderId::new("test"),
            response_text: "Fallback response".to_string(),
            model: ModelId::new("mock-model"),
        };

        let mut conversation = ConversationState::new();
        conversation.append_user("Hi there");

        let registry = ToolRegistry::new();
        let adapter = PassthroughAdapter;
        let config = SessionConfig {
            provider_id: ProviderId::new("test"),
            model: Some(ModelId::new("mock-model")),
            system_prompt: None,
            tool_policy: Default::default(),
            limits: SessionLimits::default(),
            metadata: Default::default(),
        };
        let approval = AutoApproveHandler;

        let (tx, mut rx) = crate::event::event_channel();

        let session_id = llm_core::SessionId::new("test-fallback");
        let outcome = run_streaming_turn_loop(TurnLoopContext {
            session_id: &session_id,
            client: &client,
            conversation: &mut conversation,
            tool_registry: &registry,
            tool_adapter: &adapter,
            config: &config,
            approval_handler: &approval,
            event_tx: Some(&tx),
        })
        .await
        .expect("should fall back to send_turn");

        assert_eq!(outcome.final_text, "Fallback response");
        assert_eq!(outcome.model.as_str(), "mock-model");
        assert_eq!(outcome.turns_used, 1);

        // Verify TurnCompleted was emitted even through fallback.
        drop(tx);
        let mut got_turn_completed = false;
        while let Some(event) = rx.recv().await {
            if matches!(event, SessionEvent::TurnCompleted { .. }) {
                got_turn_completed = true;
            }
        }
        assert!(got_turn_completed, "expected a TurnCompleted event");
    }
}
