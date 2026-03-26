use llm_provider_api::{ProviderToolCall, ProviderToolDescriptor, ToolSchemaAdapter};
use serde_json::{Value, json};

/// Translates tool descriptors and parses tool calls in the format expected
/// by the OpenAI Chat Completions API.
pub struct OpenAiToolFormat;

impl ToolSchemaAdapter for OpenAiToolFormat {
    /// Convert provider-agnostic tool descriptors into the OpenAI
    /// `tools` array format:
    ///
    /// ```json
    /// [
    ///   {
    ///     "type": "function",
    ///     "function": {
    ///       "name": "...",
    ///       "description": "...",
    ///       "parameters": { ... }
    ///     }
    ///   }
    /// ]
    /// ```
    fn translate_descriptors(&self, tools: &[ProviderToolDescriptor]) -> Vec<Value> {
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                })
            })
            .collect()
    }

    /// Parse tool calls from a raw OpenAI chat completion response.
    ///
    /// Expects the standard shape:
    /// ```json
    /// {
    ///   "choices": [{
    ///     "message": {
    ///       "tool_calls": [{
    ///         "id": "call_...",
    ///         "type": "function",
    ///         "function": { "name": "...", "arguments": "..." }
    ///       }]
    ///     }
    ///   }]
    /// }
    /// ```
    fn parse_tool_calls(&self, response: &Value) -> Vec<ProviderToolCall> {
        let Some(tool_calls) = response
            .pointer("/choices/0/message/tool_calls")
            .and_then(|v| v.as_array())
        else {
            return Vec::new();
        };

        tool_calls
            .iter()
            .filter_map(|tc| {
                let id = tc.get("id")?.as_str()?.to_owned();
                let func = tc.get("function")?;
                let name = func.get("name")?.as_str()?.to_owned();
                let args_str = func.get("arguments")?.as_str()?;
                let arguments: Value =
                    serde_json::from_str(args_str).unwrap_or(Value::Object(Default::default()));
                Some(ProviderToolCall {
                    id,
                    name,
                    arguments,
                })
            })
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_descriptors_produces_correct_shape() {
        let adapter = OpenAiToolFormat;
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
            },
        ];

        let result = adapter.translate_descriptors(&descriptors);
        assert_eq!(result.len(), 2);

        // First tool
        assert_eq!(result[0]["type"], "function");
        assert_eq!(result[0]["function"]["name"], "get_weather");
        assert_eq!(
            result[0]["function"]["description"],
            "Get the current weather for a location."
        );
        assert_eq!(
            result[0]["function"]["parameters"]["properties"]["location"]["type"],
            "string"
        );

        // Second tool
        assert_eq!(result[1]["function"]["name"], "search");
    }

    #[test]
    fn parse_tool_calls_extracts_from_response() {
        let adapter = OpenAiToolFormat;
        let response = json!({
            "id": "chatcmpl-abc",
            "model": "gpt-4o",
            "choices": [{
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
                                "arguments": "{\"location\":\"San Francisco\"}"
                            }
                        },
                        {
                            "id": "call_002",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": "{\"query\":\"rust async\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let calls = adapter.parse_tool_calls(&response);
        assert_eq!(calls.len(), 2);

        assert_eq!(calls[0].id, "call_001");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "San Francisco");

        assert_eq!(calls[1].id, "call_002");
        assert_eq!(calls[1].name, "search");
        assert_eq!(calls[1].arguments["query"], "rust async");
    }

    #[test]
    fn parse_tool_calls_returns_empty_for_no_tool_calls() {
        let adapter = OpenAiToolFormat;
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                }
            }]
        });

        let calls = adapter.parse_tool_calls(&response);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_tool_calls_handles_malformed_arguments() {
        let adapter = OpenAiToolFormat;
        let response = json!({
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "broken",
                            "arguments": "not valid json{"
                        }
                    }]
                }
            }]
        });

        let calls = adapter.parse_tool_calls(&response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "broken");
        // Malformed JSON falls back to an empty object.
        assert!(calls[0].arguments.is_object());
    }
}
