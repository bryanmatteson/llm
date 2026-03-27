use std::fmt;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use llm_core::{FrameworkError, Metadata, Result, ToolId};

use crate::context::ToolContext;

// ---------------------------------------------------------------------------
// ToolInfo — what the user provides
// ---------------------------------------------------------------------------

/// The human-authored metadata for a tool: its identity, names, and
/// description. The JSON Schema for the input is derived automatically
/// from the `Tool::Input` type — you do not set it here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    /// Stable framework-level identifier (e.g. `"weather"`).
    pub id: ToolId,
    /// Name used in the provider wire format (e.g. `"get_weather"`).
    pub wire_name: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Prose description shown to the model.
    pub description: String,
    /// Arbitrary key-value metadata.
    #[serde(default)]
    pub metadata: Metadata,
    /// Provider-specific top-level fields merged into the advertised tool
    /// object for providers that support custom tool properties.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extensions: Map<String, Value>,
}

impl ToolInfo {
    /// Create a new `ToolInfo` where `wire_name` defaults to the same as `id`.
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        let id_str: String = id.into();
        Self {
            wire_name: id_str.clone(),
            display_name: id_str.clone(),
            id: ToolId::new(id_str),
            description: description.into(),
            metadata: Metadata::new(),
            extensions: Map::new(),
        }
    }

    /// Set the wire name (if different from the id).
    pub fn wire_name(mut self, name: impl Into<String>) -> Self {
        self.wire_name = name.into();
        self
    }

    /// Set the display name (if different from the id).
    pub fn display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = name.into();
        self
    }

    /// Add a provider-specific top-level field to the advertised tool object.
    pub fn extension(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extensions.insert(key.into(), value);
        self
    }
}

// ---------------------------------------------------------------------------
// ToolDescriptor — full descriptor including the JSON Schema
// ---------------------------------------------------------------------------

/// Full tool descriptor including the auto-generated JSON Schema for the
/// input type. You get this from [`DynTool::descriptor()`], not by
/// constructing it manually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub id: ToolId,
    pub wire_name: String,
    pub display_name: String,
    pub description: String,
    pub parameters: Value,
    pub metadata: Metadata,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extensions: Map<String, Value>,
}

// ---------------------------------------------------------------------------
// Tool — the public, typed trait that users implement
// ---------------------------------------------------------------------------

/// A strongly-typed tool that can be invoked by an LLM.
///
/// Implement this trait to define a tool. The JSON Schema sent to the provider
/// is derived automatically from `Input` via [`schemars::JsonSchema`], and
/// serde handles serialization at the boundary.
///
/// # Example
///
/// ```ignore
/// use serde::{Deserialize, Serialize};
/// use schemars::JsonSchema;
///
/// #[derive(Debug, Deserialize, JsonSchema)]
/// struct SearchInput {
///     /// The query to search for.
///     query: String,
/// }
///
/// #[derive(Debug, Serialize)]
/// struct SearchOutput {
///     results: Vec<String>,
/// }
///
/// #[derive(Debug)]
/// struct SearchTool;
///
/// #[async_trait::async_trait]
/// impl llm_tools::Tool for SearchTool {
///     type Input = SearchInput;
///     type Output = SearchOutput;
///
///     fn info(&self) -> ToolInfo {
///         ToolInfo::new("search", "Search the web")
///     }
///
///     async fn execute(&self, input: SearchInput, _ctx: &ToolContext) -> Result<SearchOutput> {
///         Ok(SearchOutput { results: vec![format!("result for '{}'", input.query)] })
///     }
/// }
/// ```
#[async_trait]
pub trait Tool: Send + Sync + fmt::Debug {
    /// The input type the model must provide.
    type Input: DeserializeOwned + JsonSchema + Send;

    /// The output type returned to the model.
    type Output: Serialize + Send;

    /// Return the tool's metadata. The JSON Schema is derived automatically
    /// from `Input` — you only provide the identity and description.
    fn info(&self) -> ToolInfo;

    /// Execute the tool with typed input.
    async fn execute(&self, input: Self::Input, ctx: &ToolContext) -> Result<Self::Output>;
}

// ---------------------------------------------------------------------------
// DynTool — internal object-safe trait for registry / mediator dispatch
// ---------------------------------------------------------------------------

/// Object-safe trait used by [`ToolRegistry`](crate::ToolRegistry) and the
/// session mediator to dispatch tool calls through `Arc<dyn DynTool>`.
///
/// You do not implement this directly — implement [`Tool`] instead, which
/// provides a blanket implementation of `DynTool`.
#[async_trait]
pub trait DynTool: Send + Sync + fmt::Debug {
    /// Returns the full descriptor for this tool (including JSON Schema).
    fn descriptor(&self) -> ToolDescriptor;

    /// Execute the tool with raw JSON input and return raw JSON output.
    async fn invoke(&self, input: Value, context: &ToolContext) -> Result<Value>;
}

fn json_schema_for<T: JsonSchema>() -> Value {
    let schema = schemars::schema_for!(T);
    serde_json::to_value(schema).unwrap_or_else(|_| serde_json::json!({"type": "object"}))
}

#[async_trait]
impl<T> DynTool for T
where
    T: Tool,
{
    fn descriptor(&self) -> ToolDescriptor {
        let info = self.info();
        ToolDescriptor {
            id: info.id,
            wire_name: info.wire_name,
            display_name: info.display_name,
            description: info.description,
            parameters: json_schema_for::<T::Input>(),
            metadata: info.metadata,
            extensions: info.extensions,
        }
    }

    async fn invoke(&self, input: Value, context: &ToolContext) -> Result<Value> {
        let info = self.info();
        let typed_input: T::Input = serde_json::from_value(input).map_err(|e| {
            FrameworkError::validation(format!(
                "failed to deserialize input for tool '{}': {e}",
                info.wire_name
            ))
        })?;

        let output = self.execute(typed_input, context).await?;

        serde_json::to_value(output)
            .map_err(|e| FrameworkError::tool(info.id, format!("failed to serialize output: {e}")))
    }
}
