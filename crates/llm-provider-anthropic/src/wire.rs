use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Messages API request ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<WireMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    pub stream: bool,
}

// ── Wire message ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMessage {
    pub role: String,
    pub content: WireContent,
}

// ── Wire content ───────────────────────────────────────────────────

/// Content can be either a plain text string or an array of content blocks.
///
/// Uses `#[serde(untagged)]` so that `"hello"` deserializes as `Text` and
/// `[{"type":"text","text":"hello"}]` deserializes as `Blocks`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WireContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

// ── Content blocks ─────────────────────────────────────────────────

/// Individual content blocks within a message.
///
/// Uses `#[serde(tag = "type")]` so the `"type"` field determines the variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult { tool_use_id: String, content: Value },
}

// ── Messages API response ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<UsageInfo>,
}

// ── Usage info ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wire_message_text_roundtrip() {
        let msg = WireMessage {
            role: "user".into(),
            content: WireContent::Text("Hello!".into()),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let back: WireMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "user");
        match back.content {
            WireContent::Text(t) => assert_eq!(t, "Hello!"),
            _ => panic!("expected Text variant"),
        }
    }

    #[test]
    fn wire_message_blocks_roundtrip() {
        let msg = WireMessage {
            role: "assistant".into(),
            content: WireContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Let me check.".into(),
                },
                ContentBlock::ToolUse {
                    id: "toolu_01".into(),
                    name: "get_weather".into(),
                    input: json!({"location": "NYC"}),
                },
            ]),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let back: WireMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "assistant");
        match &back.content {
            WireContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    ContentBlock::Text { text } => assert_eq!(text, "Let me check."),
                    other => panic!("expected Text block, got {other:?}"),
                }
                match &blocks[1] {
                    ContentBlock::ToolUse { id, name, input } => {
                        assert_eq!(id, "toolu_01");
                        assert_eq!(name, "get_weather");
                        assert_eq!(input["location"], "NYC");
                    }
                    other => panic!("expected ToolUse block, got {other:?}"),
                }
            }
            _ => panic!("expected Blocks variant"),
        }
    }

    #[test]
    fn tool_result_block_roundtrip() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "toolu_01".into(),
            content: json!("Sunny, 72F"),
        };

        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"tool_result"#));
        let back: ContentBlock = serde_json::from_str(&json).unwrap();
        match back {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
            } => {
                assert_eq!(tool_use_id, "toolu_01");
                assert_eq!(content, json!("Sunny, 72F"));
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[test]
    fn messages_request_roundtrip() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-20250514".into(),
            max_tokens: 1024,
            system: Some("You are helpful.".into()),
            messages: vec![WireMessage {
                role: "user".into(),
                content: WireContent::Text("Hi".into()),
            }],
            tools: vec![],
            stream: false,
        };

        let json = serde_json::to_string(&req).unwrap();
        // tools should be omitted when empty
        assert!(!json.contains("\"tools\""));
        let back: MessagesRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "claude-sonnet-4-20250514");
        assert_eq!(back.max_tokens, 1024);
        assert_eq!(back.system.as_deref(), Some("You are helpful."));
        assert!(!back.stream);
    }

    #[test]
    fn messages_request_no_system_omitted() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-20250514".into(),
            max_tokens: 512,
            system: None,
            messages: vec![],
            tools: vec![],
            stream: false,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("\"system\""));
    }

    #[test]
    fn messages_response_deserialize() {
        let json = r#"{
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "model": "claude-sonnet-4-20250514",
            "content": [
                { "type": "text", "text": "Hello!" }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 25,
                "output_tokens": 10
            }
        }"#;

        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "msg_01XFDUDYJgAACzvnptvVoYEL");
        assert_eq!(resp.model, "claude-sonnet-4-20250514");
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Hello!"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        let usage = resp.usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, 25);
        assert_eq!(usage.output_tokens, 10);
    }

    #[test]
    fn messages_response_with_tool_use() {
        let json = r#"{
            "id": "msg_abc",
            "model": "claude-sonnet-4-20250514",
            "content": [
                { "type": "text", "text": "I'll check the weather." },
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 50,
                "output_tokens": 30
            }
        }"#;

        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 2);
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));

        match &resp.content[1] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "toolu_01");
                assert_eq!(name, "get_weather");
                assert_eq!(input["location"], "San Francisco");
            }
            other => panic!("expected ToolUse, got {other:?}"),
        }
    }

    #[test]
    fn messages_request_with_tools_included() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-20250514".into(),
            max_tokens: 1024,
            system: None,
            messages: vec![],
            tools: vec![json!({
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" }
                    }
                }
            })],
            stream: false,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"tools\""));
        assert!(json.contains("get_weather"));
    }
}
