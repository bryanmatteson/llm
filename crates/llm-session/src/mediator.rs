use llm_core::{ContentBlock, FrameworkError, Message, ModelId, Result, StopReason, TokenUsage};
use llm_provider_api::{LlmProviderClient, ProviderToolDescriptor, ToolSchemaAdapter, TurnRequest};
use llm_tools::{ToolApproval, ToolContext, ToolRegistry};

use crate::approval::{ApprovalHandler, ApprovalRequest, ApprovalResponse};
use crate::config::SessionConfig;
use crate::conversation::ConversationState;
use crate::event::{EventSender, SessionEvent};

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

// ---------------------------------------------------------------------------
// Turn loop
// ---------------------------------------------------------------------------

/// Execute the multi-step provider/tool turn loop.
///
/// This is the central orchestration function of `llm-session`.  It sends
/// conversation turns to the provider, dispatches tool calls (subject to the
/// configured [`ToolPolicy`](llm_tools::ToolPolicy) and
/// [`ApprovalHandler`]), appends results to the conversation, and repeats
/// until the model produces a final text response or a limit is reached.
///
/// # Arguments
///
/// * `client` - The LLM provider client to send turns to.
/// * `conversation` - Mutable conversation state; messages are appended in place as the loop progresses.
/// * `tool_registry` - Registry of available tools.
/// * `tool_adapter` - Provider-specific schema translator.
/// * `config` - Session configuration (limits, policy, etc.).
/// * `approval_handler` - Handler for obtaining human approval when required.
/// * `event_tx`         - Optional channel for emitting progress events.
#[allow(clippy::too_many_arguments)]
pub async fn run_turn_loop(
    session_id: &llm_core::SessionId,
    client: &dyn LlmProviderClient,
    conversation: &mut ConversationState,
    tool_registry: &ToolRegistry,
    tool_adapter: &dyn ToolSchemaAdapter,
    config: &SessionConfig,
    approval_handler: &dyn ApprovalHandler,
    event_tx: Option<&EventSender>,
) -> Result<TurnOutcome> {
    let tool_descriptors_json = translate_tools(tool_registry, tool_adapter);

    let mut total_tool_calls: usize = 0;
    let mut aggregated_usage = TokenUsage::default();
    // Initialized to a fallback; overwritten on each turn by the provider's
    // reported model.  The initial value is only used if the loop exits
    // without completing a single turn (which cannot happen given
    // max_turns >= 1), but we need a value to satisfy the compiler.
    let mut last_model = config
        .model
        .clone()
        .unwrap_or_else(|| ModelId::new("unknown"));
    let _ = &last_model; // suppress unused-assignment warning

    for turn_index in 0..config.limits.max_turns {
        // 1. Build the request.
        let request = build_turn_request(conversation, config, &tool_descriptors_json);

        // 2. Call the provider.
        let response = tokio::time::timeout(
            config.limits.turn_timeout,
            client.send_turn(&request),
        )
        .await
        .map_err(|_| FrameworkError::session(format!(
            "turn timed out after {:?}", config.limits.turn_timeout
        )))??;

        // Accumulate usage.
        aggregated_usage.input_tokens += response.usage.input_tokens;
        aggregated_usage.output_tokens += response.usage.output_tokens;
        last_model = response.model.clone();

        // 3. Normalize: append the provider's response messages to the
        //    conversation transcript.
        for msg in &response.messages {
            conversation.append_message(msg.clone());
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
                        event_tx,
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

                let mut tool_calls_this_turn: usize = 0;

                for (call_id, tool_name, arguments) in calls {
                    if tool_calls_this_turn >= config.limits.max_tool_calls_per_turn {
                        let err_msg = format!(
                            "Tool call limit ({}) reached; call to '{}' was skipped.",
                            config.limits.max_tool_calls_per_turn, tool_name
                        );
                        conversation.append_tool_result(&call_id, &err_msg);
                        total_tool_calls += 1;
                        continue;
                    }

                    emit(
                        event_tx,
                        SessionEvent::ToolCallRequested {
                            call_id: call_id.clone(),
                            tool_name: tool_name.clone(),
                            arguments: arguments.clone(),
                        },
                    );

                    // Look up the tool by wire name.
                    let tool = match tool_registry.get_by_wire_name(&tool_name) {
                        Some(t) => t,
                        None => {
                            // Tool not found -- inform the model.
                            let err_msg =
                                format!("Tool '{tool_name}' is not available.");
                            conversation.append_tool_result(&call_id, &err_msg);
                            emit(
                                event_tx,
                                SessionEvent::Error {
                                    message: err_msg,
                                },
                            );
                            total_tool_calls += 1;
                            tool_calls_this_turn += 1;
                            continue;
                        }
                    };

                    let descriptor = tool.descriptor();

                    // -- Policy check --
                    let approval = config.tool_policy.approval_for(&descriptor.id);

                    match approval {
                        ToolApproval::Deny => {
                            let deny_msg = format!(
                                "Tool '{tool_name}' is denied by policy."
                            );
                            conversation.append_tool_result(&call_id, &deny_msg);
                            emit(
                                event_tx,
                                SessionEvent::Error {
                                    message: deny_msg,
                                },
                            );
                            total_tool_calls += 1;
                            tool_calls_this_turn += 1;
                            continue;
                        }
                        ToolApproval::RequireConfirmation => {
                            emit(
                                event_tx,
                                SessionEvent::ToolApprovalRequired {
                                    call_id: call_id.clone(),
                                    tool_name: tool_name.clone(),
                                    arguments: arguments.clone(),
                                },
                            );

                            let approval_resp = approval_handler
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
                                    let deny_msg = reason.unwrap_or_else(|| {
                                        format!("User denied tool call '{tool_name}'.")
                                    });
                                    conversation.append_tool_result(&call_id, &deny_msg);
                                    total_tool_calls += 1;
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
                        session_id: session_id.clone(),
                        provider_id: config.provider_id.clone(),
                        model_id: last_model.clone(),
                        metadata: Default::default(),
                    };

                    let exec_result = match tokio::time::timeout(
                        config.limits.tool_timeout,
                        tool.invoke(arguments.clone(), &tool_context),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => Err(FrameworkError::tool(
                            descriptor.id.clone(),
                            format!("execution timed out after {:?}", config.limits.tool_timeout),
                        )),
                    };

                    let (content, summary) = match exec_result {
                        Ok(value) => {
                            let content_str = serde_json::to_string(&value)
                                .unwrap_or_else(|_| value.to_string());
                            let summary = content_str
                                .chars()
                                .take(120)
                                .collect::<String>();
                            (content_str, summary)
                        }
                        Err(e) => {
                            let err_str = format!("Tool execution error: {e}");
                            (err_str.clone(), err_str)
                        }
                    };

                    conversation.append_tool_result(&call_id, &content);

                    emit(
                        event_tx,
                        SessionEvent::ToolCallCompleted {
                            call_id: call_id.clone(),
                            tool_name: tool_name.clone(),
                            summary,
                        },
                    );

                    total_tool_calls += 1;
                    tool_calls_this_turn += 1;
                }

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
                    event_tx,
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
        event_tx,
        SessionEvent::TurnLimitReached {
            turns_used: config.limits.max_turns,
        },
    );

    Err(FrameworkError::session(format!(
        "turn limit reached after {} turns",
        config.limits.max_turns
    )))
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
            Err(FrameworkError::unsupported("streaming not implemented in mock"))
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
        let outcome = run_turn_loop(
            &session_id,
            &client,
            &mut conversation,
            &registry,
            &adapter,
            &config,
            &approval,
            Some(&tx),
        )
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
}
