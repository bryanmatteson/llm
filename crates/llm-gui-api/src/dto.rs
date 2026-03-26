//! Data Transfer Objects for the GUI API layer.
//!
//! Every DTO is `Serialize + Clone + Debug` so it can be sent over any
//! transport (JSON-RPC, WebSocket, IPC) without leaking internal types.

use serde::{Deserialize, Serialize};

/// Describes an LLM provider visible to the GUI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderDto {
    pub id: String,
    pub display_name: String,
    pub capabilities: Vec<String>,
}

/// A user-facing account summary for a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountDto {
    pub provider_id: String,
    pub display_name: String,
}

/// Authentication status for a single provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthStatusDto {
    pub provider_id: String,
    pub authenticated: bool,
    pub method: Option<String>,
}

/// A lightweight session summary suitable for list views.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDto {
    pub id: String,
    pub provider_id: String,
    pub model: String,
    pub message_count: usize,
}

/// Describes a tool registered in the framework.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDto {
    pub id: String,
    pub display_name: String,
    pub description: String,
}

/// A single question in a questionnaire flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionDto {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub required: bool,
}

/// An answer submitted for a questionnaire question.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerDto {
    pub question_id: String,
    pub value: serde_json::Value,
}

/// A GUI-facing event emitted during a session turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventDto {
    pub kind: String,
    pub data: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn provider_dto_serializes() {
        let dto = ProviderDto {
            id: "openai".into(),
            display_name: "OpenAI".into(),
            capabilities: vec!["OAuth".into(), "Streaming".into()],
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["id"], "openai");
        assert_eq!(json["display_name"], "OpenAI");
        assert_eq!(json["capabilities"], json!(["OAuth", "Streaming"]));
    }

    #[test]
    fn account_dto_serializes() {
        let dto = AccountDto {
            provider_id: "anthropic".into(),
            display_name: "My Anthropic Account".into(),
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["provider_id"], "anthropic");
        assert_eq!(json["display_name"], "My Anthropic Account");
    }

    #[test]
    fn auth_status_dto_serializes() {
        let dto = AuthStatusDto {
            provider_id: "openai".into(),
            authenticated: true,
            method: Some("OAuth".into()),
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["authenticated"], true);
        assert_eq!(json["method"], "OAuth");

        let dto_none = AuthStatusDto {
            provider_id: "openai".into(),
            authenticated: false,
            method: None,
        };
        let json_none = serde_json::to_value(&dto_none).unwrap();
        assert_eq!(json_none["authenticated"], false);
        assert!(json_none["method"].is_null());
    }

    #[test]
    fn session_dto_serializes() {
        let dto = SessionDto {
            id: "sess-001".into(),
            provider_id: "openai".into(),
            model: "gpt-4o".into(),
            message_count: 12,
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["id"], "sess-001");
        assert_eq!(json["message_count"], 12);
    }

    #[test]
    fn tool_dto_serializes() {
        let dto = ToolDto {
            id: "echo".into(),
            display_name: "Echo".into(),
            description: "Echoes input back.".into(),
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["id"], "echo");
        assert_eq!(json["description"], "Echoes input back.");
    }

    #[test]
    fn question_dto_serializes() {
        let dto = QuestionDto {
            id: "q1".into(),
            label: "Pick a language".into(),
            kind: "Choice".into(),
            required: true,
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["kind"], "Choice");
        assert_eq!(json["required"], true);
    }

    #[test]
    fn answer_dto_serializes() {
        let dto = AnswerDto {
            question_id: "q1".into(),
            value: json!("en"),
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["question_id"], "q1");
        assert_eq!(json["value"], "en");
    }

    #[test]
    fn event_dto_serializes() {
        let dto = EventDto {
            kind: "assistant_delta".into(),
            data: json!({"text": "Hello!"}),
        };
        let json = serde_json::to_value(&dto).unwrap();
        assert_eq!(json["kind"], "assistant_delta");
        assert_eq!(json["data"]["text"], "Hello!");
    }

    #[test]
    fn dtos_are_clone_and_debug() {
        let provider = ProviderDto {
            id: "p".into(),
            display_name: "P".into(),
            capabilities: vec![],
        };
        let cloned = provider.clone();
        assert_eq!(format!("{cloned:?}"), format!("{:?}", provider));

        let event = EventDto {
            kind: "test".into(),
            data: json!(null),
        };
        let _cloned_event = event.clone();
        let _debug = format!("{event:?}");
    }

    #[test]
    fn dto_roundtrip_json() {
        let original = SessionDto {
            id: "s1".into(),
            provider_id: "anthropic".into(),
            model: "claude-3".into(),
            message_count: 5,
        };
        let json_str = serde_json::to_string(&original).unwrap();
        let restored: SessionDto = serde_json::from_str(&json_str).unwrap();
        assert_eq!(restored.id, original.id);
        assert_eq!(restored.provider_id, original.provider_id);
        assert_eq!(restored.model, original.model);
        assert_eq!(restored.message_count, original.message_count);
    }
}
