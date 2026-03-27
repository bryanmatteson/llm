use llm_core::Message;
use llm_tools::ToolInfo;
use serde_json::{Map, Value, json};

pub const TOOL_SEARCH_TOOL_REGEX_20251119: &str = "tool_search_tool_regex_20251119";
pub const TOOL_SEARCH_TOOL_BM25_20251119: &str = "tool_search_tool_bm25_20251119";
pub const MESSAGE_CACHE_CONTROL_METADATA_KEY: &str = "anthropic.cache_control_json";

pub fn cache_control_ephemeral(ttl: Option<&str>) -> Value {
    let mut value = json!({ "type": "ephemeral" });
    if let Some(ttl) = ttl {
        value["ttl"] = Value::String(ttl.to_string());
    }
    value
}

pub fn tool_search_regex_tool() -> Value {
    json!({
        "type": TOOL_SEARCH_TOOL_REGEX_20251119,
        "name": "tool_search_tool_regex",
    })
}

pub fn tool_search_bm25_tool() -> Value {
    json!({
        "type": TOOL_SEARCH_TOOL_BM25_20251119,
        "name": "tool_search_tool_bm25",
    })
}

pub fn tool_reference_block(tool_name: impl Into<String>) -> Value {
    json!({
        "type": "tool_reference",
        "tool_name": tool_name.into(),
    })
}

pub fn system_text_block(text: impl Into<String>) -> Value {
    json!({
        "type": "text",
        "text": text.into(),
    })
}

pub fn cacheable_system_text_block(text: impl Into<String>, ttl: Option<&str>) -> Value {
    json!({
        "type": "text",
        "text": text.into(),
        "cache_control": cache_control_ephemeral(ttl),
    })
}

#[derive(Debug, Clone, Default)]
pub struct ClearToolUsesConfig {
    trigger_input_tokens: Option<u64>,
    keep_tool_uses: Option<u64>,
    clear_at_least_input_tokens: Option<u64>,
    clear_tool_inputs: Option<Value>,
    exclude_tools: Vec<String>,
}

impl ClearToolUsesConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn trigger_input_tokens(mut self, value: u64) -> Self {
        self.trigger_input_tokens = Some(value);
        self
    }

    pub fn keep_tool_uses(mut self, value: u64) -> Self {
        self.keep_tool_uses = Some(value);
        self
    }

    pub fn clear_at_least_input_tokens(mut self, value: u64) -> Self {
        self.clear_at_least_input_tokens = Some(value);
        self
    }

    pub fn clear_all_tool_inputs(mut self) -> Self {
        self.clear_tool_inputs = Some(Value::Bool(true));
        self
    }

    pub fn clear_tool_inputs_for<I, S>(mut self, tool_names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.clear_tool_inputs = Some(Value::Array(
            tool_names
                .into_iter()
                .map(|name| Value::String(name.into()))
                .collect(),
        ));
        self
    }

    pub fn exclude_tool(mut self, tool_name: impl Into<String>) -> Self {
        self.exclude_tools.push(tool_name.into());
        self
    }

    pub fn to_edit_value(&self) -> Value {
        let mut edit = Map::new();
        edit.insert(
            "type".into(),
            Value::String("clear_tool_uses_20250919".into()),
        );
        if let Some(value) = self.trigger_input_tokens {
            edit.insert(
                "trigger".into(),
                json!({"type": "input_tokens", "value": value}),
            );
        }
        if let Some(value) = self.keep_tool_uses {
            edit.insert("keep".into(), json!({"type": "tool_uses", "value": value}));
        }
        if let Some(value) = self.clear_at_least_input_tokens {
            edit.insert(
                "clear_at_least".into(),
                json!({"type": "input_tokens", "value": value}),
            );
        }
        if let Some(value) = &self.clear_tool_inputs {
            edit.insert("clear_tool_inputs".into(), value.clone());
        }
        if !self.exclude_tools.is_empty() {
            edit.insert("exclude_tools".into(), json!(self.exclude_tools));
        }
        Value::Object(edit)
    }
}

pub fn context_management(edit: Value) -> Value {
    json!({
        "edits": [edit],
    })
}

pub trait AnthropicToolInfoExt {
    fn allow_callers<I, S>(self, callers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>;

    fn eager_input_streaming(self) -> Self;

    fn defer_loading(self) -> Self;

    fn cache_ephemeral(self, ttl: Option<&str>) -> Self;
}

impl AnthropicToolInfoExt for ToolInfo {
    fn allow_callers<I, S>(self, callers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extension(
            "allowed_callers",
            Value::Array(
                callers
                    .into_iter()
                    .map(|caller| Value::String(caller.into()))
                    .collect(),
            ),
        )
    }

    fn eager_input_streaming(self) -> Self {
        self.extension("eager_input_streaming", Value::Bool(true))
    }

    fn defer_loading(self) -> Self {
        self.extension("defer_loading", Value::Bool(true))
    }

    fn cache_ephemeral(self, ttl: Option<&str>) -> Self {
        self.extension("cache_control", cache_control_ephemeral(ttl))
    }
}

pub trait AnthropicMessageExt {
    fn cache_ephemeral(self, ttl: Option<&str>) -> Self;
}

impl AnthropicMessageExt for Message {
    fn cache_ephemeral(mut self, ttl: Option<&str>) -> Self {
        self.metadata.insert(
            MESSAGE_CACHE_CONTROL_METADATA_KEY.into(),
            serde_json::to_string(&cache_control_ephemeral(ttl))
                .unwrap_or_else(|_| r#"{"type":"ephemeral"}"#.to_string()),
        );
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_tool_uses_config_serializes_expected_shape() {
        let edit = ClearToolUsesConfig::new()
            .trigger_input_tokens(500)
            .keep_tool_uses(2)
            .clear_at_least_input_tokens(100)
            .clear_all_tool_inputs()
            .exclude_tool("search")
            .to_edit_value();

        assert_eq!(edit["type"], "clear_tool_uses_20250919");
        assert_eq!(edit["trigger"]["value"], 500);
        assert_eq!(edit["keep"]["value"], 2);
        assert_eq!(edit["clear_at_least"]["value"], 100);
        assert_eq!(edit["clear_tool_inputs"], true);
        assert_eq!(edit["exclude_tools"][0], "search");
    }

    #[test]
    fn anthropic_message_ext_adds_cache_control_metadata() {
        let msg = Message::user("hi").cache_ephemeral(Some("1h"));
        let cache = msg
            .metadata
            .get(MESSAGE_CACHE_CONTROL_METADATA_KEY)
            .unwrap();
        assert!(cache.contains("1h"));
    }

    #[test]
    fn tool_reference_block_uses_expected_shape() {
        let block = tool_reference_block("get_weather");
        assert_eq!(block["type"], "tool_reference");
        assert_eq!(block["tool_name"], "get_weather");
    }
}
