use serde::{Deserialize, Serialize};

// ── Chat Completions request ────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

// ── Wire message ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMessage {
    pub role: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,

    /// Present on messages with role `"tool"` to identify which tool call
    /// this result corresponds to.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Optional `name` field (used by some older endpoints or tool results).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

// ── Tool calls ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireToolCall {
    pub id: String,

    #[serde(rename = "type")]
    pub type_field: String,

    pub function: WireFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireFunctionCall {
    pub name: String,
    pub arguments: String,
}

// ── Chat Completions response ───────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<WireChoice>,
    #[serde(default)]
    pub usage: Option<WireUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireChoice {
    pub index: u32,
    pub message: WireMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

// ── Models endpoint response ────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    pub data: Vec<WireModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireModel {
    pub id: String,
    #[serde(default)]
    pub owned_by: Option<String>,
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_message_roundtrip() {
        let msg = WireMessage {
            role: "user".into(),
            content: Some("Hello!".into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let back: WireMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "user");
        assert_eq!(back.content.as_deref(), Some("Hello!"));
        assert!(back.tool_calls.is_none());
    }

    #[test]
    fn wire_tool_call_roundtrip() {
        let tc = WireToolCall {
            id: "call_abc123".into(),
            type_field: "function".into(),
            function: WireFunctionCall {
                name: "get_weather".into(),
                arguments: r#"{"location":"NYC"}"#.into(),
            },
        };

        let json = serde_json::to_string(&tc).unwrap();
        assert!(json.contains(r#""type":"function""#));
        let back: WireToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "call_abc123");
        assert_eq!(back.type_field, "function");
        assert_eq!(back.function.name, "get_weather");
    }

    #[test]
    fn chat_completion_request_roundtrip() {
        let req = ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![WireMessage {
                role: "user".into(),
                content: Some("Hi".into()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            temperature: Some(0.7),
            max_tokens: Some(1024),
        };

        let json = serde_json::to_string(&req).unwrap();
        let back: ChatCompletionRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "gpt-4o");
        assert_eq!(back.messages.len(), 1);
        assert_eq!(back.temperature, Some(0.7));
        assert_eq!(back.max_tokens, Some(1024));
    }

    #[test]
    fn chat_completion_response_deserialize() {
        let json = r#"{
            "id": "chatcmpl-abc",
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let resp: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "chatcmpl-abc");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        let usage = resp.usage.as_ref().unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn chat_completion_response_with_tool_calls() {
        let json = r#"{
            "id": "chatcmpl-xyz",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_001",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "{\"location\":\"NYC\"}"
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30
            }
        }"#;

        let resp: ChatCompletionResponse = serde_json::from_str(json).unwrap();
        let msg = &resp.choices[0].message;
        assert!(msg.content.is_none());
        let tool_calls = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_001");
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn model_list_response_deserialize() {
        let json = r#"{
            "data": [
                { "id": "gpt-4o", "owned_by": "openai" },
                { "id": "gpt-4o-mini", "owned_by": "openai" }
            ]
        }"#;

        let resp: ModelListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].id, "gpt-4o");
        assert_eq!(resp.data[1].id, "gpt-4o-mini");
    }

    #[test]
    fn optional_fields_omitted_in_serialization() {
        let req = ChatCompletionRequest {
            model: "gpt-4o".into(),
            messages: vec![],
            tools: None,
            temperature: None,
            max_tokens: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tools"));
        assert!(!json.contains("temperature"));
        assert!(!json.contains("max_tokens"));
    }
}
