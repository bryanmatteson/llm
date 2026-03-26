use std::collections::HashMap;
use std::sync::Arc;

use llm_core::ToolId;

use crate::tool::{DynTool, ToolDescriptor};

/// A registry that maps [`ToolId`]s to tool implementations.
///
/// Tools are stored behind `Arc<dyn DynTool>` so they can be shared across
/// async tasks. Any type that implements [`Tool`](crate::Tool) automatically
/// implements `DynTool` and can be registered here.
#[derive(Debug, Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<ToolId, Arc<dyn DynTool>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool. If a tool with the same ID already exists it will be
    /// replaced.
    pub fn register(&mut self, tool: Arc<dyn DynTool>) {
        let descriptor = tool.descriptor();
        self.tools.insert(descriptor.id, tool);
    }

    /// Look up a tool by its [`ToolId`].
    pub fn get(&self, id: &ToolId) -> Option<Arc<dyn DynTool>> {
        self.tools.get(id).cloned()
    }

    /// Look up a tool by its wire-format name (the name sent to the provider
    /// API).
    pub fn get_by_wire_name(&self, name: &str) -> Option<Arc<dyn DynTool>> {
        self.tools
            .values()
            .find(|t| t.descriptor().wire_name == name)
            .cloned()
    }

    /// Return descriptors for every registered tool, sorted by tool ID for
    /// deterministic output.
    pub fn all_descriptors(&self) -> Vec<ToolDescriptor> {
        let mut descriptors: Vec<ToolDescriptor> =
            self.tools.values().map(|t| t.descriptor()).collect();
        descriptors.sort_by(|a, b| a.id.cmp(&b.id));
        descriptors
    }

    /// Return the IDs of every registered tool, sorted.
    pub fn tool_ids(&self) -> Vec<ToolId> {
        let mut ids: Vec<ToolId> = self.tools.keys().cloned().collect();
        ids.sort();
        ids
    }
}
