use std::fmt;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use llm_core::{Metadata, Result, ToolId};

use crate::context::ToolContext;

/// Describes a tool's interface: its identity, wire-format name, human-readable
/// name, description, and expected input parameters as a JSON Schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    /// Unique framework-level identifier for this tool.
    pub id: ToolId,
    /// Name used in the provider API wire format (e.g. sent to OpenAI as the
    /// function name).
    pub wire_name: String,
    /// Human-readable display name.
    pub display_name: String,
    /// A prose description of what this tool does.
    pub description: String,
    /// JSON Schema describing the expected input parameters.
    pub parameters: Value,
    /// Arbitrary key-value metadata attached to this tool.
    pub metadata: Metadata,
}

/// An executable tool that can be invoked by a model.
///
/// Implementors provide a [`ToolDescriptor`] and an async `execute` method that
/// receives JSON input and returns a JSON output or an error.
#[async_trait]
pub trait Tool: Send + Sync + fmt::Debug {
    /// Returns the descriptor for this tool.
    fn descriptor(&self) -> ToolDescriptor;

    /// Execute the tool with the given JSON `input` and invocation context.
    ///
    /// Returns a JSON value on success, or a [`FrameworkError`] on failure.
    async fn execute(&self, input: Value, context: &ToolContext) -> Result<Value>;
}
