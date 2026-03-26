use std::sync::Arc;

use llm_tools::{Tool, ToolDescriptor, ToolRegistry};

/// Service for managing the set of available tools.
pub struct ToolService {
    tool_registry: Arc<ToolRegistry>,
}

impl ToolService {
    /// Create a new `ToolService` wrapping the given registry.
    pub fn new(tool_registry: Arc<ToolRegistry>) -> Self {
        Self { tool_registry }
    }

    /// Return descriptors for every registered tool, sorted by tool id.
    pub fn list_tools(&self) -> Vec<ToolDescriptor> {
        self.tool_registry.all_descriptors()
    }

    /// Register a new tool into the shared registry.
    ///
    /// Because `ToolRegistry` requires `&mut self` for `register`, this
    /// method needs mutable access to the inner `Arc`. It will succeed only
    /// if no other reference to the registry exists (i.e.
    /// [`Arc::get_mut`] returns `Some`).
    ///
    /// In practice, tools should be registered during the build phase (via
    /// [`AppBuilder`](crate::builder::AppBuilder)) before the registry is
    /// shared. This method exists as a convenience for late-registered tools
    /// in tests or plugin scenarios.
    pub fn register_tool(&mut self, tool: Arc<dyn Tool>) {
        if let Some(registry) = Arc::get_mut(&mut self.tool_registry) {
            registry.register(tool);
        }
    }

    /// Obtain a shared reference to the underlying [`ToolRegistry`].
    pub fn registry(&self) -> &Arc<ToolRegistry> {
        &self.tool_registry
    }
}
