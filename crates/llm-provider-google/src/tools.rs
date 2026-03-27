use llm_provider_api::{ProviderToolCall, ProviderToolDescriptor, ToolSchemaAdapter};
use serde_json::{Value, json};

/// Translates tool descriptors and parses tool calls in the format expected
/// by the Google Gemini API.
///
/// Google wraps all function declarations in a single `tools` object:
/// ```json
/// [
///   {
///     "function_declarations": [
///       { "name": "...", "description": "...", "parameters": { ... } }
///     ]
///   }
/// ]
/// ```
pub struct GoogleToolFormat;

impl ToolSchemaAdapter for GoogleToolFormat {
    /// Convert provider-agnostic tool descriptors into the Gemini
    /// `tools` array format.
    ///
    /// Google wraps all declarations into a single tool object.
    fn translate_descriptors(&self, tools: &[ProviderToolDescriptor]) -> Vec<Value> {
        if tools.is_empty() {
            return Vec::new();
        }

        let declarations: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();

        // Google expects a single-element array with all declarations grouped.
        vec![json!({
            "function_declarations": declarations,
        })]
    }

    /// Parse tool calls from a raw Gemini generateContent response.
    ///
    /// Extracts `functionCall` parts from the first candidate:
    /// ```json
    /// {
    ///   "candidates": [{
    ///     "content": {
    ///       "parts": [
    ///         { "functionCall": { "name": "...", "args": { ... } } }
    ///       ]
    ///     }
    ///   }]
    /// }
    /// ```
    fn parse_tool_calls(&self, response: &Value) -> Vec<ProviderToolCall> {
        let Some(parts) = response
            .pointer("/candidates/0/content/parts")
            .and_then(|v| v.as_array())
        else {
            return Vec::new();
        };

        let mut calls = Vec::new();
        for (idx, part) in parts.iter().enumerate() {
            if let Some(fc) = part.get("functionCall") {
                let name = fc
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_owned();
                let arguments = fc
                    .get("args")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));

                calls.push(ProviderToolCall {
                    // Gemini does not provide call IDs; synthesize one.
                    id: format!("gemini_call_{idx}"),
                    name,
                    arguments,
                });
            }
        }

        calls
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_descriptors_produces_google_format() {
        let adapter = GoogleToolFormat;
        let descriptors = vec![
            ProviderToolDescriptor {
                name: "get_weather".into(),
                description: "Get the current weather for a location.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" }
                    },
                    "required": ["location"]
                }),
                extensions: Default::default(),
            },
            ProviderToolDescriptor {
                name: "search".into(),
                description: "Search the web.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string" }
                    },
                    "required": ["query"]
                }),
                extensions: Default::default(),
            },
        ];

        let result = adapter.translate_descriptors(&descriptors);
        // Google wraps all declarations in a single object.
        assert_eq!(result.len(), 1);

        let decls = result[0]["function_declarations"].as_array().unwrap();
        assert_eq!(decls.len(), 2);

        assert_eq!(decls[0]["name"], "get_weather");
        assert_eq!(
            decls[0]["description"],
            "Get the current weather for a location."
        );
        assert_eq!(
            decls[0]["parameters"]["properties"]["location"]["type"],
            "string"
        );

        assert_eq!(decls[1]["name"], "search");
    }

    #[test]
    fn translate_descriptors_returns_empty_for_no_tools() {
        let adapter = GoogleToolFormat;
        let result = adapter.translate_descriptors(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn parse_tool_calls_extracts_from_response() {
        let adapter = GoogleToolFormat;
        let response = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Let me check."},
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "San Francisco"}
                            }
                        },
                        {
                            "functionCall": {
                                "name": "search",
                                "args": {"query": "rust async"}
                            }
                        }
                    ]
                },
                "finishReason": "STOP"
            }]
        });

        let calls = adapter.parse_tool_calls(&response);
        assert_eq!(calls.len(), 2);

        assert_eq!(calls[0].id, "gemini_call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "San Francisco");

        assert_eq!(calls[1].id, "gemini_call_2");
        assert_eq!(calls[1].name, "search");
        assert_eq!(calls[1].arguments["query"], "rust async");
    }

    #[test]
    fn parse_tool_calls_returns_empty_for_text_only() {
        let adapter = GoogleToolFormat;
        let response = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello!"}]
                }
            }]
        });

        let calls = adapter.parse_tool_calls(&response);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_tool_calls_returns_empty_for_no_candidates() {
        let adapter = GoogleToolFormat;
        let response = json!({
            "candidates": []
        });

        let calls = adapter.parse_tool_calls(&response);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_tool_calls_handles_missing_args() {
        let adapter = GoogleToolFormat;
        let response = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "functionCall": {
                            "name": "no_args_tool"
                        }
                    }]
                }
            }]
        });

        let calls = adapter.parse_tool_calls(&response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "no_args_tool");
        assert!(calls[0].arguments.is_object());
    }
}
