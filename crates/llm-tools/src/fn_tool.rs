use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use llm_core::{FrameworkError, Result, ToolId};

use crate::context::ToolContext;
use crate::tool::{DynTool, ToolDescriptor, ToolInfo};

/// Type alias for the handler function signature used by [`FnTool`].
///
/// The handler receives a shared reference to application state `S` and
/// a JSON `Value` of arguments, and returns a future producing a
/// `Result<Value, E>` where `E` converts into `FrameworkError`.
pub type HandlerFn<S, E> = for<'a> fn(
    &'a S,
    Value,
) -> Pin<Box<dyn Future<Output = std::result::Result<Value, E>> + Send + 'a>>;

/// A [`DynTool`] built from a function pointer and captured application state.
///
/// The error type `E` can be any type that implements `Into<FrameworkError>`,
/// letting you use your own typed errors in tool handlers while the `DynTool`
/// boundary converts them automatically.
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
/// async fn search(state: &AppState, args: Value) -> Result<Value, MyError> {
///     let query = args["query"].as_str().unwrap_or("");
///     let results = state.db.search(query).await?;
///     Ok(json!({ "results": results }))
/// }
///
/// let state = Arc::new(AppState { db });
/// let tool = FnTool::new(
///     ToolInfo::new("search", "Search the database"),
///     json!({"type": "object", "properties": {"query": {"type": "string"}}}),
///     state,
///     |s, args| Box::pin(search(s, args)),
/// );
///
/// let mut registry = ToolRegistry::new();
/// registry.register(Arc::new(tool));
/// ```
pub struct FnTool<S, E = FrameworkError>
where
    S: Send + Sync + 'static,
    E: Into<FrameworkError> + Send + 'static,
{
    info: ToolInfo,
    parameters: Value,
    state: Arc<S>,
    handler: HandlerFn<S, E>,
}

impl<S, E> FnTool<S, E>
where
    S: Send + Sync + 'static,
    E: Into<FrameworkError> + Send + 'static,
{
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
        handler: HandlerFn<S, E>,
    ) -> Self {
        Self {
            info,
            parameters,
            state,
            handler,
        }
    }
}

impl<S, E> fmt::Debug for FnTool<S, E>
where
    S: Send + Sync + 'static,
    E: Into<FrameworkError> + Send + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FnTool")
            .field("id", &self.info.id)
            .field("wire_name", &self.info.wire_name)
            .finish()
    }
}

#[async_trait]
impl<S, E> DynTool for FnTool<S, E>
where
    S: Send + Sync + 'static,
    E: Into<FrameworkError> + Send + 'static,
{
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
        (self.handler)(&self.state, input)
            .await
            .map_err(Into::into)
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

    async fn count_handler(
        state: &Counter,
        args: Value,
    ) -> std::result::Result<Value, FrameworkError> {
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
        async fn failing(
            _: &(),
            _args: Value,
        ) -> std::result::Result<Value, FrameworkError> {
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

    // Test with a custom error type that implements Into<FrameworkError>
    #[derive(Debug)]
    struct CustomError(String);

    impl From<CustomError> for FrameworkError {
        fn from(e: CustomError) -> Self {
            FrameworkError::tool(ToolId::new("custom"), e.0)
        }
    }

    async fn custom_err_handler(
        _: &(),
        _args: Value,
    ) -> std::result::Result<Value, CustomError> {
        Err(CustomError("custom error".to_string()))
    }

    #[tokio::test]
    async fn fn_tool_converts_custom_error() {
        let tool = FnTool::new(
            ToolInfo::new("custom", "Custom error test"),
            json!({"type": "object"}),
            Arc::new(()),
            |s, args| Box::pin(custom_err_handler(s, args)),
        );

        let ctx = make_context();
        let err = tool.invoke(json!({}), &ctx).await.unwrap_err();
        assert!(err.to_string().contains("custom error"));
    }

    #[test]
    fn fn_tool_debug_shows_id() {
        let tool: FnTool<(), FrameworkError> = FnTool::new(
            ToolInfo::new("debug_test", "test"),
            json!({}),
            Arc::new(()),
            |_, args| Box::pin(async move { Ok(args) }),
        );
        let debug = format!("{tool:?}");
        assert!(debug.contains("debug_test"));
    }
}
