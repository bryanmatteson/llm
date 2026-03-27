use serde::{Deserialize, Serialize};

use serde_json::{Map, Value};

/// A lightweight tool descriptor for provider-level interchange.
///
/// This avoids depending on `llm-tools` while providing the essential
/// information a provider needs to advertise tool schemas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderToolDescriptor {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extensions: Map<String, Value>,
}

/// A lightweight representation of a parsed tool call returned by a provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Adapter trait for translating tool schemas and parsing tool calls
/// in a provider-specific manner.
///
/// Both inputs and outputs use `serde_json::Value` so that this trait
/// can live in `llm-provider-api` without depending on `llm-tools`.
pub trait ToolSchemaAdapter: Send + Sync {
    /// Translate a slice of provider-agnostic tool descriptors into the
    /// JSON format expected by a specific provider API.
    fn translate_descriptors(&self, tools: &[ProviderToolDescriptor]) -> Vec<serde_json::Value>;

    /// Parse provider-specific tool call representations from a raw
    /// JSON response into `ProviderToolCall` values.
    fn parse_tool_calls(&self, response: &serde_json::Value) -> Vec<ProviderToolCall>;
}
