use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── GenerateContent request ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentRequest {
    pub contents: Vec<WireContent>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<WireContent>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<WireTool>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
}

// ── Wire content ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<WirePart>,
}

// ── Wire part ───────────────────────────────────────────────────────

/// A single part within a Gemini content block.
///
/// Parts can contain text, function calls, or function responses.
/// Uses optional fields rather than an enum to match the Gemini wire format
/// where any combination of fields may be present.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct WirePart {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<WireFunctionCall>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_response: Option<WireFunctionResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireFunctionCall {
    pub name: String,
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireFunctionResponse {
    pub name: String,
    pub response: Value,
}

// ── Tools ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WireTool {
    pub function_declarations: Vec<FunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

// ── Generation config ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

// ── GenerateContent response ────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateContentResponse {
    #[serde(default)]
    pub candidates: Vec<Candidate>,

    #[serde(default)]
    pub usage_metadata: Option<UsageMetadata>,

    /// Present when the prompt itself is blocked by safety filters.
    #[serde(default)]
    pub prompt_feedback: Option<PromptFeedback>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub content: WireContent,

    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    #[serde(default)]
    pub prompt_token_count: u64,
    #[serde(default)]
    pub candidates_token_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptFeedback {
    #[serde(default)]
    pub block_reason: Option<String>,
}

// ── Models list response ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelListResponse {
    #[serde(default)]
    pub models: Vec<WireModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WireModel {
    /// Full resource name, e.g. `"models/gemini-2.5-flash"`.
    pub name: String,
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub input_token_limit: Option<u64>,
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wire_content_text_roundtrip() {
        let content = WireContent {
            role: Some("user".into()),
            parts: vec![WirePart {
                text: Some("Hello!".into()),
                ..Default::default()
            }],
        };

        let json = serde_json::to_string(&content).unwrap();
        let back: WireContent = serde_json::from_str(&json).unwrap();
        assert_eq!(back.role.as_deref(), Some("user"));
        assert_eq!(back.parts.len(), 1);
        assert_eq!(back.parts[0].text.as_deref(), Some("Hello!"));
    }

    #[test]
    fn wire_part_function_call_roundtrip() {
        let part = WirePart {
            function_call: Some(WireFunctionCall {
                name: "get_weather".into(),
                args: json!({"location": "NYC"}),
            }),
            ..Default::default()
        };

        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("functionCall"));
        let back: WirePart = serde_json::from_str(&json).unwrap();
        let fc = back.function_call.unwrap();
        assert_eq!(fc.name, "get_weather");
        assert_eq!(fc.args["location"], "NYC");
    }

    #[test]
    fn wire_part_function_response_roundtrip() {
        let part = WirePart {
            function_response: Some(WireFunctionResponse {
                name: "get_weather".into(),
                response: json!({"temperature": 72}),
            }),
            ..Default::default()
        };

        let json = serde_json::to_string(&part).unwrap();
        assert!(json.contains("functionResponse"));
        let back: WirePart = serde_json::from_str(&json).unwrap();
        let fr = back.function_response.unwrap();
        assert_eq!(fr.name, "get_weather");
        assert_eq!(fr.response["temperature"], 72);
    }

    #[test]
    fn generate_content_request_roundtrip() {
        let req = GenerateContentRequest {
            contents: vec![WireContent {
                role: Some("user".into()),
                parts: vec![WirePart {
                    text: Some("Hi".into()),
                    ..Default::default()
                }],
            }],
            system_instruction: Some(WireContent {
                role: None,
                parts: vec![WirePart {
                    text: Some("You are helpful.".into()),
                    ..Default::default()
                }],
            }),
            tools: vec![],
            generation_config: Some(GenerationConfig {
                max_output_tokens: Some(1024),
                temperature: Some(0.7),
            }),
        };

        let json = serde_json::to_string(&req).unwrap();
        // tools should be omitted when empty
        assert!(!json.contains("\"tools\""));
        let back: GenerateContentRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.contents.len(), 1);
        assert!(back.system_instruction.is_some());
        let config = back.generation_config.unwrap();
        assert_eq!(config.max_output_tokens, Some(1024));
        assert_eq!(config.temperature, Some(0.7));
    }

    #[test]
    fn generate_content_request_no_optionals() {
        let req = GenerateContentRequest {
            contents: vec![],
            system_instruction: None,
            tools: vec![],
            generation_config: None,
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("systemInstruction"));
        assert!(!json.contains("tools"));
        assert!(!json.contains("generationConfig"));
    }

    #[test]
    fn generate_content_response_text_deserialize() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello there!"}]
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5
            }
        }"#;

        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.candidates.len(), 1);
        assert_eq!(resp.candidates[0].content.parts[0].text.as_deref(), Some("Hello there!"));
        assert_eq!(resp.candidates[0].finish_reason.as_deref(), Some("STOP"));
        let usage = resp.usage_metadata.as_ref().unwrap();
        assert_eq!(usage.prompt_token_count, 10);
        assert_eq!(usage.candidates_token_count, 5);
    }

    #[test]
    fn generate_content_response_with_function_call() {
        let json = r#"{
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Let me check."},
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "NYC"}
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP"
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 20,
                "candidatesTokenCount": 15
            }
        }"#;

        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        let parts = &resp.candidates[0].content.parts;
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].text.as_deref(), Some("Let me check."));
        let fc = parts[1].function_call.as_ref().unwrap();
        assert_eq!(fc.name, "get_weather");
        assert_eq!(fc.args["location"], "NYC");
    }

    #[test]
    fn generate_content_response_with_prompt_feedback() {
        let json = r#"{
            "candidates": [],
            "promptFeedback": {
                "blockReason": "SAFETY"
            }
        }"#;

        let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
        assert!(resp.candidates.is_empty());
        let feedback = resp.prompt_feedback.as_ref().unwrap();
        assert_eq!(feedback.block_reason.as_deref(), Some("SAFETY"));
    }

    #[test]
    fn wire_tool_roundtrip() {
        let tool = WireTool {
            function_declarations: vec![FunctionDeclaration {
                name: "search".into(),
                description: "Search the web.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                }),
            }],
        };

        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("functionDeclarations"));
        let back: WireTool = serde_json::from_str(&json).unwrap();
        assert_eq!(back.function_declarations.len(), 1);
        assert_eq!(back.function_declarations[0].name, "search");
    }

    #[test]
    fn generation_config_roundtrip() {
        let config = GenerationConfig {
            max_output_tokens: Some(2048),
            temperature: Some(0.5),
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("maxOutputTokens"));
        let back: GenerationConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_output_tokens, Some(2048));
        assert_eq!(back.temperature, Some(0.5));
    }

    #[test]
    fn generation_config_omits_none_fields() {
        let config = GenerationConfig {
            max_output_tokens: None,
            temperature: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("maxOutputTokens"));
        assert!(!json.contains("temperature"));
    }

    #[test]
    fn model_list_response_deserialize() {
        let json = r#"{
            "models": [
                {
                    "name": "models/gemini-2.5-flash",
                    "displayName": "Gemini 2.5 Flash",
                    "inputTokenLimit": 1048576
                },
                {
                    "name": "models/gemini-2.5-pro",
                    "displayName": "Gemini 2.5 Pro"
                }
            ]
        }"#;

        let resp: ModelListResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.models.len(), 2);
        assert_eq!(resp.models[0].name, "models/gemini-2.5-flash");
        assert_eq!(resp.models[0].display_name.as_deref(), Some("Gemini 2.5 Flash"));
        assert_eq!(resp.models[0].input_token_limit, Some(1048576));
        assert_eq!(resp.models[1].name, "models/gemini-2.5-pro");
        assert!(resp.models[1].input_token_limit.is_none());
    }

    #[test]
    fn usage_metadata_defaults_to_zero() {
        let json = r#"{}"#;
        let usage: UsageMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(usage.prompt_token_count, 0);
        assert_eq!(usage.candidates_token_count, 0);
    }

    #[test]
    fn wire_content_no_role_roundtrip() {
        // system_instruction content has no role
        let content = WireContent {
            role: None,
            parts: vec![WirePart {
                text: Some("Be helpful.".into()),
                ..Default::default()
            }],
        };

        let json = serde_json::to_string(&content).unwrap();
        assert!(!json.contains("\"role\""));
        let back: WireContent = serde_json::from_str(&json).unwrap();
        assert!(back.role.is_none());
        assert_eq!(back.parts[0].text.as_deref(), Some("Be helpful."));
    }
}
