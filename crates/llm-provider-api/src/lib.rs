pub mod adapter;
pub mod client;
pub mod event;

pub use adapter::{ProviderToolCall, ProviderToolDescriptor, ToolSchemaAdapter};
pub use client::{LlmProviderClient, TurnRequest, TurnResponse};
pub use event::ProviderEvent;

#[cfg(test)]
mod tests {
    use llm_core::{Message, ModelId, StopReason, TokenUsage};

    use crate::{TurnRequest, TurnResponse};

    #[test]
    fn construct_turn_request() {
        let request = TurnRequest {
            system_prompt: Some("You are a helpful assistant.".to_string()),
            messages: vec![Message::user("Hello, world!")],
            tools: vec![],
            model: Some(ModelId::new("gpt-4o")),
            max_tokens: Some(1024),
            temperature: Some(0.7),
        };

        assert_eq!(
            request.system_prompt.as_deref(),
            Some("You are a helpful assistant.")
        );
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.model.as_ref().unwrap().as_str(), "gpt-4o");
        assert_eq!(request.max_tokens, Some(1024));
    }

    #[test]
    fn construct_turn_response() {
        let response = TurnResponse {
            messages: vec![Message::assistant("Hello! How can I help?")],
            stop_reason: StopReason::EndTurn,
            model: ModelId::new("gpt-4o"),
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 8,
            },
        };

        assert_eq!(response.messages.len(), 1);
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.model.as_str(), "gpt-4o");
        assert_eq!(response.usage.total(), 18);
    }
}
