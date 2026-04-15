use serde::{Deserialize, Serialize};
use serde_json::Value;

fn is_false(value: &bool) -> bool {
    !*value
}

// ── Messages API request ───────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<ContainerRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_management: Option<Value>,
    pub messages: Vec<WireMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub stream: bool,
    #[serde(flatten, default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra_body: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

// ── Wire message ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireMessage {
    pub role: String,
    pub content: WireContent,
}

// ── Wire content ───────────────────────────────────────────────────

/// Content can be either a plain text string or an array of raw content
/// blocks. Using raw JSON values allows us to preserve Anthropic-specific
/// blocks such as `server_tool_use` without needing a closed enum.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WireContent {
    Text(String),
    Blocks(Vec<Value>),
}

// ── Messages API response ──────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessagesResponse {
    pub id: String,
    pub model: String,
    pub content: Vec<Value>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<UsageInfo>,
    #[serde(default)]
    pub container: Option<ContainerResponse>,
    #[serde(default)]
    pub context_management: Option<Value>,
}

// ── Usage info ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerResponse {
    pub id: String,
    #[serde(default)]
    pub expires_at: Option<String>,
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
                json!({
                    "type": "text",
                    "text": "Let me check."
                }),
                json!({
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"location": "NYC"}
                }),
            ]),
        };

        let json = serde_json::to_string(&msg).unwrap();
        let back: WireMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role, "assistant");
        match &back.content {
            WireContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert_eq!(blocks[0]["type"], "text");
                assert_eq!(blocks[0]["text"], "Let me check.");
                assert_eq!(blocks[1]["type"], "tool_use");
                assert_eq!(blocks[1]["id"], "toolu_01");
                assert_eq!(blocks[1]["name"], "get_weather");
                assert_eq!(blocks[1]["input"]["location"], "NYC");
            }
            _ => panic!("expected Blocks variant"),
        }
    }

    #[test]
    fn messages_request_roundtrip() {
        let req = MessagesRequest {
            model: "claude-sonnet-4-20250514".into(),
            max_tokens: 1024,
            temperature: Some(0.2),
            system: Some(json!("You are helpful.")),
            container: Some(ContainerRequest {
                id: Some("container_01".into()),
            }),
            cache_control: Some(json!({"type": "ephemeral"})),
            context_management: Some(json!({"edits": []})),
            messages: vec![WireMessage {
                role: "user".into(),
                content: WireContent::Text("Hi".into()),
            }],
            tools: vec![],
            stream: false,
            extra_body: serde_json::Map::from_iter([(
                "tool_choice".into(),
                json!({"type": "auto"}),
            )]),
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("\"tools\""));
        let back: MessagesRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "claude-sonnet-4-20250514");
        assert_eq!(back.max_tokens, 1024);
        assert_eq!(back.temperature, Some(0.2));
        assert_eq!(back.system, Some(json!("You are helpful.")));
        assert_eq!(
            back.container.as_ref().and_then(|c| c.id.as_deref()),
            Some("container_01")
        );
        assert_eq!(back.cache_control, Some(json!({"type": "ephemeral"})));
        assert_eq!(back.context_management, Some(json!({"edits": []})));
        assert_eq!(back.extra_body["tool_choice"]["type"], "auto");
        assert!(!back.stream);
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
            },
            "container": {
                "id": "container_01",
                "expires_at": "2099-01-01T00:00:00Z"
            }
        }"#;

        let resp: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, "msg_01XFDUDYJgAACzvnptvVoYEL");
        assert_eq!(resp.model, "claude-sonnet-4-20250514");
        assert_eq!(resp.content.len(), 1);
        assert_eq!(resp.content[0]["type"], "text");
        assert_eq!(resp.content[0]["text"], "Hello!");
        assert_eq!(resp.stop_reason.as_deref(), Some("end_turn"));
        let usage = resp.usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, 25);
        assert_eq!(usage.output_tokens, 10);
        assert_eq!(
            resp.container.as_ref().map(|c| c.id.as_str()),
            Some("container_01")
        );
    }
}
