pub mod transport;

use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub use llm_tools::{DynTool, FnTool, ToolContext, ToolDescriptor, ToolInfo, ToolRegistry};

const JSONRPC_VERSION: &str = "2.0";
const MCP_PROTOCOL_VERSION: &str = "2025-03-26";

// ── MCP wire-format tool definition ────────────────────────────────────

/// MCP wire-format tool definition, as returned by `tools/list`.
///
/// This is the serialization shape required by the MCP protocol.  It is
/// derived from a [`ToolDescriptor`] via the [`From`] impl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

impl From<&ToolDescriptor> for McpToolDefinition {
    fn from(desc: &ToolDescriptor) -> Self {
        Self {
            name: desc.wire_name.clone(),
            description: desc.description.clone(),
            input_schema: desc.parameters.clone(),
        }
    }
}

impl From<ToolDescriptor> for McpToolDefinition {
    fn from(desc: ToolDescriptor) -> Self {
        Self::from(&desc)
    }
}

// ── McpServer ──────────────────────────────────────────────────────────

/// A generic MCP (Model Context Protocol) server.
///
/// Handles JSON-RPC 2.0 message routing for the MCP protocol, dispatching
/// `tools/call` requests to a [`ToolRegistry`].
#[derive(Clone)]
pub struct McpServer {
    registry: Arc<ToolRegistry>,
    context: Arc<ToolContext>,
    server_name: String,
    server_version: String,
}

impl McpServer {
    /// Create a new MCP server.
    ///
    /// - `registry` – the tool registry containing all available tools.
    /// - `context`  – default context passed to tools on invocation.
    /// - `server_name` – the name reported in `initialize` (e.g. `"stag"`).
    /// - `server_version` – the version reported in `initialize`.
    pub fn new(
        registry: ToolRegistry,
        context: ToolContext,
        server_name: impl Into<String>,
        server_version: impl Into<String>,
    ) -> Self {
        Self {
            registry: Arc::new(registry),
            context: Arc::new(context),
            server_name: server_name.into(),
            server_version: server_version.into(),
        }
    }

    /// Return a reference to the tool registry.
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Process a single JSON-RPC message and return an optional response.
    ///
    /// Notifications (messages without an `id`) return `None`.
    pub async fn handle_message(&self, message: Value) -> Option<Value> {
        if !message.is_object() {
            return Some(jsonrpc_error(
                Value::Null,
                -32600,
                "invalid request payload",
            ));
        }

        let method = message
            .get("method")
            .and_then(Value::as_str)
            .map(str::to_string);
        let id = message.get("id").cloned();
        let params = message.get("params").cloned().unwrap_or(Value::Null);

        let Some(method) = method else {
            if id.is_some() {
                return Some(jsonrpc_error(
                    id.unwrap_or(Value::Null),
                    -32600,
                    "invalid request",
                ));
            }
            return None;
        };

        let response = self.handle_method(id.clone(), &method, params).await;
        if id.is_some() { Some(response) } else { None }
    }

    async fn handle_method(&self, id: Option<Value>, method: &str, params: Value) -> Value {
        let request_id = id.unwrap_or(Value::Null);
        match method {
            "initialize" => {
                let result = json!({
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": { "listChanged": false }
                    },
                    "serverInfo": {
                        "name": self.server_name,
                        "version": self.server_version,
                    }
                });
                jsonrpc_result(request_id, result)
            }
            "ping" => jsonrpc_result(request_id, json!({})),
            "shutdown" => jsonrpc_result(request_id, Value::Null),
            "notifications/initialized" => jsonrpc_result(request_id, Value::Null),
            "tools/list" => {
                let defs: Vec<McpToolDefinition> = self
                    .registry
                    .all_descriptors()
                    .into_iter()
                    .map(McpToolDefinition::from)
                    .collect();
                let result = json!({ "tools": defs });
                jsonrpc_result(request_id, result)
            }
            "tools/call" => {
                let response = self.handle_tool_call(params).await;
                jsonrpc_result(request_id, response)
            }
            _ => jsonrpc_error(request_id, -32601, "method not found"),
        }
    }

    async fn handle_tool_call(&self, params: Value) -> Value {
        let parsed = serde_json::from_value::<ToolCallParams>(params);
        let params = match parsed {
            Ok(params) => params,
            Err(error) => {
                return json!({
                    "isError": true,
                    "content": [{
                        "type": "text",
                        "text": format!("invalid tool params: {error}"),
                    }],
                    "structuredContent": { "error": "invalid tool params" }
                });
            }
        };

        let tool = match self.registry.get_by_wire_name(&params.name) {
            Some(t) => t,
            None => {
                return json!({
                    "isError": true,
                    "content": [{
                        "type": "text",
                        "text": format!("unknown tool '{}'", params.name),
                    }],
                    "structuredContent": { "error": format!("unknown tool '{}'", params.name) }
                });
            }
        };

        match tool.invoke(params.arguments, &self.context).await {
            Ok(payload) => {
                let text = serde_json::to_string_pretty(&payload)
                    .unwrap_or_else(|error| format!("tool response serialization failed: {error}"));
                json!({
                    "isError": false,
                    "content": [{
                        "type": "text",
                        "text": text,
                    }],
                    "structuredContent": payload,
                })
            }
            Err(error) => json!({
                "isError": true,
                "content": [{
                    "type": "text",
                    "text": format!("{error:#}"),
                }],
                "structuredContent": { "error": error.to_string() }
            }),
        }
    }
}

// ── JSON-RPC helpers ───────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ToolCallParams {
    name: String,
    #[serde(default)]
    arguments: Value,
}

/// Build a JSON-RPC 2.0 success response.
#[must_use]
pub fn jsonrpc_result(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id,
        "result": result,
    })
}

/// Build a JSON-RPC 2.0 error response.
#[must_use]
pub fn jsonrpc_error(id: Value, code: i32, message: &str) -> Value {
    json!({
        "jsonrpc": JSONRPC_VERSION,
        "id": id,
        "error": {
            "code": code,
            "message": message,
        }
    })
}

/// Parse a single line of JSON text.
pub fn parse_json_line(input: &str) -> Result<Value> {
    serde_json::from_str::<Value>(input).map_err(|e| anyhow::anyhow!("failed to parse JSON: {e}"))
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use serde_json::json;

    use llm_core::{ModelId, ProviderId, SessionId};
    use llm_tools::test_utils::EchoTool;

    use super::*;

    fn make_context() -> ToolContext {
        ToolContext {
            session_id: SessionId::new("test"),
            provider_id: ProviderId::new("test"),
            model_id: ModelId::new("test"),
            metadata: BTreeMap::new(),
        }
    }

    fn make_server() -> McpServer {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        McpServer::new(registry, make_context(), "test-server", "0.1.0")
    }

    #[tokio::test]
    async fn initialize_returns_server_info() {
        let server = make_server();
        let response = server
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {}
            }))
            .await
            .expect("response");
        assert_eq!(response["result"]["serverInfo"]["name"], "test-server");
        assert_eq!(response["result"]["serverInfo"]["version"], "0.1.0");
        assert_eq!(response["result"]["protocolVersion"], "2025-03-26");
    }

    #[tokio::test]
    async fn ping_returns_empty_result() {
        let server = make_server();
        let response = server
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": "p1",
                "method": "ping"
            }))
            .await
            .expect("response");
        assert_eq!(response["id"], "p1");
        assert!(response["result"].is_object());
    }

    #[tokio::test]
    async fn tools_list_returns_registered_tools() {
        let server = make_server();
        let response = server
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            }))
            .await
            .expect("response");
        let tools = response["result"]["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "echo");
    }

    #[tokio::test]
    async fn tool_call_success() {
        let server = make_server();
        let response = server
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "echo",
                    "arguments": { "message": "hello" }
                }
            }))
            .await
            .expect("response");
        assert_eq!(response["result"]["isError"], false);
        assert_eq!(response["result"]["structuredContent"]["echoed"], "hello");
    }

    #[tokio::test]
    async fn tool_call_unknown_tool_returns_error() {
        let server = make_server();
        let response = server
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "nonexistent",
                    "arguments": {}
                }
            }))
            .await
            .expect("response");
        assert_eq!(response["result"]["isError"], true);
        let text = response["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("unknown tool"));
    }

    #[tokio::test]
    async fn invalid_request_returns_error() {
        let server = make_server();
        let response = server
            .handle_message(json!("not an object"))
            .await
            .expect("response");
        assert_eq!(response["error"]["code"], -32600);
    }

    #[tokio::test]
    async fn notification_without_id_returns_none() {
        let server = make_server();
        let response = server
            .handle_message(json!({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }))
            .await;
        assert!(response.is_none());
    }

    #[tokio::test]
    async fn unknown_method_returns_not_found() {
        let server = make_server();
        let response = server
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "unknown/thing"
            }))
            .await
            .expect("response");
        assert_eq!(response["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn shutdown_returns_null_result() {
        let server = make_server();
        let response = server
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 99,
                "method": "shutdown"
            }))
            .await
            .expect("response");
        assert!(response["result"].is_null());
    }

    #[test]
    fn parse_json_line_valid() {
        let result = parse_json_line(r#"{"method":"ping"}"#).unwrap();
        assert_eq!(result["method"], "ping");
    }

    #[test]
    fn parse_json_line_invalid() {
        assert!(parse_json_line("not json").is_err());
    }

    #[tokio::test]
    async fn invalid_tool_params_returns_error() {
        let server = make_server();
        let response = server
            .handle_message(json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": "not an object"
            }))
            .await
            .expect("response");
        assert_eq!(response["result"]["isError"], true);
        let text = response["result"]["content"][0]["text"]
            .as_str()
            .expect("error text");
        assert!(text.contains("invalid tool params"));
    }

    #[test]
    fn mcp_tool_definition_from_descriptor() {
        let desc = ToolDescriptor {
            id: llm_core::ToolId::new("search"),
            wire_name: "web_search".to_string(),
            display_name: "Web Search".to_string(),
            description: "Search the web".to_string(),
            parameters: json!({"type": "object"}),
            metadata: BTreeMap::new(),
            extensions: serde_json::Map::new(),
        };
        let def = McpToolDefinition::from(&desc);
        assert_eq!(def.name, "web_search");
        assert_eq!(def.description, "Search the web");
    }
}
