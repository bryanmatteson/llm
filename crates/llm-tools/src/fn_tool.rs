use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use llm_core::{Result, ToolId};

use crate::context::ToolContext;
use crate::tool::{DynTool, ToolDescriptor, ToolInfo};

/// Type alias for the handler function signature used by [`FnTool`].
///
/// The handler receives a shared reference to application state `S` and
/// a JSON `Value` of arguments, and returns a future producing a
/// `Result<Value, FrameworkError>`.
pub type HandlerFn<S> = for<'a> fn(
    &'a S,
    Value,
) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>>;

/// A [`DynTool`] built from a function pointer and captured application state.
///
/// This lets you expose existing `async fn(state, args) -> Result<Value>`
/// functions as tools without implementing the full typed [`Tool`](crate::Tool)
/// trait.
///
/// # Example
///
/// ```ignore
/// use llm_tools::{FnTool, ToolInfo, ToolRegistry};
/// use llm_core::{FrameworkError, ToolId};
/// use serde_json::{Value, json};
///
/// struct AppState { db: Database }
///
/// async fn search(state: &AppState, args: Value) -> llm_core::Result<Value> {
///     let query = args["query"].as_str().unwrap_or("");
///     let results = state.db.search(query).await
///         .map_err(|e| FrameworkError::tool(ToolId::new("search"), e.to_string()))?;
///     Ok(json!({ "results": results }))
/// }
///
/// let state = Arc::new(AppState { db });
/// let tool = FnTool::new(
///     ToolInfo::new("search", "Search the database"),
///     json!({"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
///     state,
///     |s, args| Box::pin(search(s, args)),
/// );
///
/// let mut registry = ToolRegistry::new();
/// registry.register(Arc::new(tool));
/// ```
pub struct FnTool<S: Send + Sync + 'static> {
    info: ToolInfo,
    parameters: Value,
    state: Arc<S>,
    handler: HandlerFn<S>,
}

impl<S: Send + Sync + 'static> FnTool<S> {
    /// Create a new function-based tool.
    ///
    /// - `info` – the tool's identity and description.
    /// - `parameters` – JSON Schema for the input (used in `tools/list`).
    /// - `state` – shared application state passed to the handler on each call.
    /// - `handler` – the async function to invoke.
    pub fn new(
        info: ToolInfo,
        parameters: Value,
        state: Arc<S>,
        handler: HandlerFn<S>,
    ) -> Self {
        Self {
            info,
            parameters,
            state,
            handler,
        }
    }
}

impl<S: Send + Sync + 'static> std::fmt::Debug for FnTool<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnTool")
            .field("id", &self.info.id)
            .field("wire_name", &self.info.wire_name)
            .finish()
    }
}

#[async_trait]
impl<S: Send + Sync + 'static> DynTool for FnTool<S> {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            id: self.info.id.clone(),
            wire_name: self.info.wire_name.clone(),
            display_name: self.info.display_name.clone(),
            description: self.info.description.clone(),
            parameters: self.parameters.clone(),
            metadata: self.info.metadata.clone(),
        }
    }

    async fn invoke(&self, input: Value, _ctx: &ToolContext) -> Result<Value> {
        (self.handler)(&self.state, input).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use serde_json::json;

    use llm_core::{FrameworkError, ModelId, ProviderId, SessionId};

    use super::*;

    struct Counter {
        label: String,
    }

    async fn count_handler(state: &Counter, args: Value) -> Result<Value> {
        let n = args["n"].as_u64().unwrap_or(1);
        Ok(json!({ "label": state.label, "count": n }))
    }

    fn make_context() -> ToolContext {
        ToolContext {
            session_id: SessionId::new("test"),
            provider_id: ProviderId::new("test"),
            model_id: ModelId::new("test"),
            metadata: BTreeMap::new(),
        }
    }

    #[tokio::test]
    async fn fn_tool_invokes_handler() {
        let state = Arc::new(Counter {
            label: "hits".to_string(),
        });
        let tool = FnTool::new(
            ToolInfo::new("counter", "Count things"),
            json!({"type": "object", "properties": {"n": {"type": "integer"}}}),
            state,
            |s, args| Box::pin(count_handler(s, args)),
        );

        let ctx = make_context();
        let result = tool.invoke(json!({"n": 42}), &ctx).await.unwrap();
        assert_eq!(result["label"], "hits");
        assert_eq!(result["count"], 42);
    }

    #[tokio::test]
    async fn fn_tool_descriptor_matches_info() {
        let state = Arc::new(Counter {
            label: "x".to_string(),
        });
        let tool = FnTool::new(
            ToolInfo::new("counter", "Count things").display_name("Counter Tool"),
            json!({"type": "object"}),
            state,
            |s, args| Box::pin(count_handler(s, args)),
        );

        let desc = tool.descriptor();
        assert_eq!(desc.id, ToolId::new("counter"));
        assert_eq!(desc.wire_name, "counter");
        assert_eq!(desc.display_name, "Counter Tool");
        assert_eq!(desc.description, "Count things");
    }

    #[tokio::test]
    async fn fn_tool_propagates_errors() {
        async fn failing(_: &(), _args: Value) -> Result<Value> {
            Err(FrameworkError::tool(
                ToolId::new("fail"),
                "intentional failure",
            ))
        }

        let tool = FnTool::new(
            ToolInfo::new("fail", "Always fails"),
            json!({"type": "object"}),
            Arc::new(()),
            |s, args| Box::pin(failing(s, args)),
        );

        let ctx = make_context();
        let err = tool.invoke(json!({}), &ctx).await.unwrap_err();
        assert!(err.to_string().contains("intentional failure"));
    }

    #[test]
    fn fn_tool_debug_shows_id() {
        let tool: FnTool<()> = FnTool::new(
            ToolInfo::new("debug_test", "test"),
            json!({}),
            Arc::new(()),
            |_, args| Box::pin(async move { Ok(args) }),
        );
        let debug = format!("{tool:?}");
        assert!(debug.contains("debug_test"));
    }
}
