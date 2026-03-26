use llm_provider_api::{ProviderToolCall, ProviderToolDescriptor, ToolSchemaAdapter};
use serde_json::{Value, json};

/// Translates tool descriptors and parses tool calls in the format expected
/// by the Anthropic Messages API.
///
/// Anthropic uses a flat tool object (no `"function"` wrapper):
/// ```json
/// {
///   "name": "...",
///   "description": "...",
///   "input_schema": { ... }
/// }
/// ```
pub struct AnthropicToolFormat;

impl ToolSchemaAdapter for AnthropicToolFormat {
    /// Convert provider-agnostic tool descriptors into the Anthropic
    /// `tools` array format:
    ///
    /// ```json
    /// [
    ///   {
    ///     "name": "...",
    ///     "description": "...",
    ///     "input_schema": { ... }
    ///   }
    /// ]
    /// ```
    fn translate_descriptors(&self, tools: &[ProviderToolDescriptor]) -> Vec<Value> {
        tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                })
            })
            .collect()
    }

    /// Parse tool calls from a raw Anthropic Messages API response.
    ///
    /// Extracts `tool_use` content blocks from the response:
    /// ```json
    /// {
    ///   "content": [
    ///     { "type": "tool_use", "id": "...", "name": "...", "input": { ... } }
    ///   ]
    /// }
    /// ```
    fn parse_tool_calls(&self, response: &Value) -> Vec<ProviderToolCall> {
        let Some(content) = response.get("content").and_then(|v| v.as_array()) else {
            return Vec::new();
        };

        content
            .iter()
            .filter_map(|block| {
                let block_type = block.get("type")?.as_str()?;
                if block_type != "tool_use" {
                    return None;
                }
                let id = block.get("id")?.as_str()?.to_owned();
                let name = block.get("name")?.as_str()?.to_owned();
                let arguments = block
                    .get("input")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default()));
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
    fn translate_descriptors_produces_anthropic_format() {
        let adapter = AnthropicToolFormat;
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

        // Anthropic format: no "type":"function" wrapper
        assert!(result[0].get("type").is_none());
        assert_eq!(result[0]["name"], "get_weather");
        assert_eq!(
            result[0]["description"],
            "Get the current weather for a location."
        );
        // Uses "input_schema" not "parameters"
        assert_eq!(
            result[0]["input_schema"]["properties"]["location"]["type"],
            "string"
        );
        assert!(result[0].get("parameters").is_none());

        assert_eq!(result[1]["name"], "search");
    }

    #[test]
    fn parse_tool_calls_extracts_from_response() {
        let adapter = AnthropicToolFormat;
        let response = json!({
            "id": "msg_abc",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {
                    "type": "text",
                    "text": "Let me check the weather."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"}
                },
                {
                    "type": "tool_use",
                    "id": "toolu_02",
                    "name": "search",
                    "input": {"query": "rust async"}
                }
            ],
            "stop_reason": "tool_use"
        });

        let calls = adapter.parse_tool_calls(&response);
        assert_eq!(calls.len(), 2);

        assert_eq!(calls[0].id, "toolu_01");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments["location"], "San Francisco");

        assert_eq!(calls[1].id, "toolu_02");
        assert_eq!(calls[1].name, "search");
        assert_eq!(calls[1].arguments["query"], "rust async");
    }

    #[test]
    fn parse_tool_calls_returns_empty_for_text_only() {
        let adapter = AnthropicToolFormat;
        let response = json!({
            "content": [
                {
                    "type": "text",
                    "text": "Hello!"
                }
            ]
        });

        let calls = adapter.parse_tool_calls(&response);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_tool_calls_returns_empty_for_missing_content() {
        let adapter = AnthropicToolFormat;
        let response = json!({
            "id": "msg_abc",
            "model": "claude-sonnet-4-20250514"
        });

        let calls = adapter.parse_tool_calls(&response);
        assert!(calls.is_empty());
    }

    #[test]
    fn parse_tool_calls_handles_missing_input() {
        let adapter = AnthropicToolFormat;
        let response = json!({
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "no_args_tool"
                }
            ]
        });

        let calls = adapter.parse_tool_calls(&response);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "no_args_tool");
        assert!(calls[0].arguments.is_object());
    }
}
