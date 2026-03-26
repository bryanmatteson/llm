//! Adapts session-layer events into transport-neutral [`EventDto`]s.
//!
//! The [`SessionEventAdapter`] converts [`llm_session::SessionEvent`] variants
//! into a flat `{ kind, data }` shape that any GUI transport can forward
//! without knowing about internal event enums.

use crate::dto::EventDto;

/// Stateless adapter that converts a serialisable session event value into an
/// [`EventDto`].
///
/// Because `llm-gui-api` does not depend on `llm-session` directly, the
/// adapter works with a `serde_json::Value` representation of the event. The
/// caller is responsible for serialising the concrete `SessionEvent` before
/// handing it to [`adapt_event`].
pub struct SessionEventAdapter;

impl SessionEventAdapter {
    /// Convert a JSON-serialised session event into an [`EventDto`].
    ///
    /// The incoming `event` is expected to be the `serde_json::Value`
    /// produced by `serde_json::to_value(&session_event)`. The function
    /// inspects the top-level tag to derive the `kind` field and passes
    /// through the variant payload as `data`.
    ///
    /// If the value cannot be interpreted (e.g. it is not an object or has
    /// no recognisable tag), a fallback `"unknown"` kind is produced.
    pub fn adapt_event(event: &serde_json::Value) -> EventDto {
        // SessionEvent is serialised with serde's default externally-tagged
        // representation:  `{ "VariantName": { ...fields... } }`
        if let Some(obj) = event.as_object() {
            if let Some((variant_name, payload)) = obj.iter().next() {
                return EventDto {
                    kind: camel_to_snake(variant_name),
                    data: payload.clone(),
                };
            }
        }

        // Fallback for unexpected shapes.
        EventDto {
            kind: "unknown".into(),
            data: event.clone(),
        }
    }
}

/// Convert a `PascalCase` or `camelCase` variant name to `snake_case`.
///
/// This keeps DTO `kind` values idiomatic for JSON consumers
/// (e.g. `"AssistantDelta"` -> `"assistant_delta"`).
fn camel_to_snake(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 4);
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(ch.to_ascii_lowercase());
        } else {
            result.push(ch);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn adapt_assistant_delta() {
        let event = json!({
            "AssistantDelta": { "text": "Hello" }
        });
        let dto = SessionEventAdapter::adapt_event(&event);
        assert_eq!(dto.kind, "assistant_delta");
        assert_eq!(dto.data["text"], "Hello");
    }

    #[test]
    fn adapt_tool_call_requested() {
        let event = json!({
            "ToolCallRequested": {
                "call_id": "c1",
                "tool_name": "echo",
                "arguments": { "message": "hi" }
            }
        });
        let dto = SessionEventAdapter::adapt_event(&event);
        assert_eq!(dto.kind, "tool_call_requested");
        assert_eq!(dto.data["call_id"], "c1");
        assert_eq!(dto.data["tool_name"], "echo");
    }

    #[test]
    fn adapt_tool_call_completed() {
        let event = json!({
            "ToolCallCompleted": {
                "call_id": "c1",
                "tool_name": "echo",
                "summary": "done"
            }
        });
        let dto = SessionEventAdapter::adapt_event(&event);
        assert_eq!(dto.kind, "tool_call_completed");
        assert_eq!(dto.data["summary"], "done");
    }

    #[test]
    fn adapt_turn_completed() {
        let event = json!({
            "TurnCompleted": {
                "text": "Final answer",
                "model": "gpt-4o",
                "usage": { "input_tokens": 10, "output_tokens": 20 }
            }
        });
        let dto = SessionEventAdapter::adapt_event(&event);
        assert_eq!(dto.kind, "turn_completed");
        assert_eq!(dto.data["text"], "Final answer");
    }

    #[test]
    fn adapt_error_event() {
        let event = json!({
            "Error": { "message": "something went wrong" }
        });
        let dto = SessionEventAdapter::adapt_event(&event);
        assert_eq!(dto.kind, "error");
        assert_eq!(dto.data["message"], "something went wrong");
    }

    #[test]
    fn adapt_turn_limit_reached() {
        let event = json!({
            "TurnLimitReached": { "turns_used": 10 }
        });
        let dto = SessionEventAdapter::adapt_event(&event);
        assert_eq!(dto.kind, "turn_limit_reached");
        assert_eq!(dto.data["turns_used"], 10);
    }

    #[test]
    fn adapt_tool_approval_required() {
        let event = json!({
            "ToolApprovalRequired": {
                "call_id": "c2",
                "tool_name": "dangerous_tool",
                "arguments": {}
            }
        });
        let dto = SessionEventAdapter::adapt_event(&event);
        assert_eq!(dto.kind, "tool_approval_required");
        assert_eq!(dto.data["tool_name"], "dangerous_tool");
    }

    #[test]
    fn adapt_unknown_shape() {
        let event = json!("just a string");
        let dto = SessionEventAdapter::adapt_event(&event);
        assert_eq!(dto.kind, "unknown");
    }

    #[test]
    fn adapt_empty_object() {
        let event = json!({});
        let dto = SessionEventAdapter::adapt_event(&event);
        assert_eq!(dto.kind, "unknown");
    }

    #[test]
    fn camel_to_snake_cases() {
        assert_eq!(super::camel_to_snake("AssistantDelta"), "assistant_delta");
        assert_eq!(super::camel_to_snake("TurnCompleted"), "turn_completed");
        assert_eq!(
            super::camel_to_snake("ToolCallRequested"),
            "tool_call_requested"
        );
        assert_eq!(super::camel_to_snake("Error"), "error");
        assert_eq!(super::camel_to_snake("already_snake"), "already_snake");
        assert_eq!(super::camel_to_snake("A"), "a");
        assert_eq!(super::camel_to_snake(""), "");
    }
}
