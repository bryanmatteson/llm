pub mod context;
pub mod invocation;
pub mod policy;
pub mod registry;
pub mod tool;
pub mod validation;

pub use context::ToolContext;
pub use invocation::{ToolCall, ToolResult};
pub use policy::{ToolApproval, ToolPolicy, ToolPolicyRule};
pub use registry::ToolRegistry;
pub use tool::{Tool, ToolDescriptor};
pub use validation::validate_tool_input;

// ---------------------------------------------------------------------------
// Test-utilities feature: EchoTool
// ---------------------------------------------------------------------------

/// A trivial tool that echoes its input, intended for integration tests.
///
/// Only available when the `test-utils` feature is enabled.
#[cfg(feature = "test-utils")]
pub mod test_utils {
    use std::collections::BTreeMap;

    use async_trait::async_trait;
    use serde_json::{Value, json};

    use llm_core::{Result, ToolId};

    use crate::context::ToolContext;
    use crate::tool::{Tool, ToolDescriptor};

    /// A minimal tool that accepts `{"message": "<string>"}` and returns
    /// `{"echoed": "<string>"}`.
    #[derive(Debug)]
    pub struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor {
                id: ToolId::new("echo"),
                wire_name: "echo".to_owned(),
                display_name: "Echo".to_owned(),
                description: "Echoes the input message back to the caller.".to_owned(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to echo."
                        }
                    },
                    "required": ["message"]
                }),
                metadata: BTreeMap::new(),
            }
        }

        async fn execute(&self, input: Value, _context: &ToolContext) -> Result<Value> {
            let message = input
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            Ok(json!({ "echoed": message }))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use async_trait::async_trait;
    use serde_json::{Value, json};

    use llm_core::{ModelId, ProviderId, Result, SessionId, ToolId};

    use crate::context::ToolContext;
    use crate::policy::{ToolApproval, ToolPolicy, ToolPolicyRule};
    use crate::registry::ToolRegistry;
    use crate::tool::{Tool, ToolDescriptor};
    use crate::validation::validate_tool_input;

    // -- helpers -----------------------------------------------------------

    /// Minimal tool used by registry / validation tests.
    #[derive(Debug)]
    struct StubTool {
        id: &'static str,
        wire: &'static str,
    }

    #[async_trait]
    impl Tool for StubTool {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor {
                id: ToolId::new(self.id),
                wire_name: self.wire.to_owned(),
                display_name: self.id.to_owned(),
                description: format!("Stub tool {}", self.id),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "x": { "type": "string" }
                    },
                    "required": ["x"]
                }),
                metadata: BTreeMap::new(),
            }
        }

        async fn execute(&self, _input: Value, _ctx: &ToolContext) -> Result<Value> {
            Ok(json!({"ok": true}))
        }
    }

    #[cfg_attr(not(feature = "test-utils"), allow(dead_code))]
    fn make_context() -> ToolContext {
        ToolContext {
            session_id: SessionId::new("test-session"),
            provider_id: ProviderId::new("test-provider"),
            model_id: ModelId::new("test-model"),
            metadata: BTreeMap::new(),
        }
    }

    // -- ToolRegistry ------------------------------------------------------

    #[test]
    fn registry_register_and_get() {
        let mut reg = ToolRegistry::new();
        let tool: Arc<dyn Tool> = Arc::new(StubTool { id: "a", wire: "wire_a" });
        reg.register(tool);

        assert!(reg.get(&ToolId::new("a")).is_some());
        assert!(reg.get(&ToolId::new("nonexistent")).is_none());
    }

    #[test]
    fn registry_get_by_wire_name() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(StubTool { id: "a", wire: "wire_a" }));
        reg.register(Arc::new(StubTool { id: "b", wire: "wire_b" }));

        let found = reg.get_by_wire_name("wire_b");
        assert!(found.is_some());
        assert_eq!(found.unwrap().descriptor().id, ToolId::new("b"));

        assert!(reg.get_by_wire_name("no_such_wire").is_none());
    }

    #[test]
    fn registry_all_descriptors_sorted() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(StubTool { id: "z_tool", wire: "wz" }));
        reg.register(Arc::new(StubTool { id: "a_tool", wire: "wa" }));

        let descs = reg.all_descriptors();
        assert_eq!(descs.len(), 2);
        assert_eq!(descs[0].id, ToolId::new("a_tool"));
        assert_eq!(descs[1].id, ToolId::new("z_tool"));
    }

    #[test]
    fn registry_tool_ids() {
        let mut reg = ToolRegistry::new();
        reg.register(Arc::new(StubTool { id: "c", wire: "wc" }));
        reg.register(Arc::new(StubTool { id: "a", wire: "wa" }));

        let ids = reg.tool_ids();
        assert_eq!(ids, vec![ToolId::new("a"), ToolId::new("c")]);
    }

    // -- ToolPolicy --------------------------------------------------------

    #[test]
    fn policy_default_approval() {
        let policy = ToolPolicy {
            default_approval: ToolApproval::RequireConfirmation,
            rules: vec![],
        };
        assert_eq!(
            policy.approval_for(&ToolId::new("anything")),
            ToolApproval::RequireConfirmation
        );
    }

    #[test]
    fn policy_rule_override() {
        let policy = ToolPolicy {
            default_approval: ToolApproval::Auto,
            rules: vec![
                ToolPolicyRule {
                    tool_id: ToolId::new("dangerous"),
                    approval: ToolApproval::Deny,
                    max_calls_per_session: None,
                },
                ToolPolicyRule {
                    tool_id: ToolId::new("sensitive"),
                    approval: ToolApproval::RequireConfirmation,
                    max_calls_per_session: Some(5),
                },
            ],
        };

        assert_eq!(policy.approval_for(&ToolId::new("dangerous")), ToolApproval::Deny);
        assert_eq!(
            policy.approval_for(&ToolId::new("sensitive")),
            ToolApproval::RequireConfirmation
        );
        // Fallback to default
        assert_eq!(policy.approval_for(&ToolId::new("normal")), ToolApproval::Auto);
    }

    #[test]
    fn policy_is_allowed() {
        let policy = ToolPolicy {
            default_approval: ToolApproval::Auto,
            rules: vec![ToolPolicyRule {
                tool_id: ToolId::new("blocked"),
                approval: ToolApproval::Deny,
                max_calls_per_session: None,
            }],
        };

        assert!(policy.is_allowed(&ToolId::new("ok_tool")));
        assert!(!policy.is_allowed(&ToolId::new("blocked")));
        // RequireConfirmation is still "allowed" (just needs confirmation)
        let policy2 = ToolPolicy {
            default_approval: ToolApproval::RequireConfirmation,
            rules: vec![],
        };
        assert!(policy2.is_allowed(&ToolId::new("anything")));
    }

    #[test]
    fn policy_serde_roundtrip() {
        let policy = ToolPolicy {
            default_approval: ToolApproval::Auto,
            rules: vec![ToolPolicyRule {
                tool_id: ToolId::new("echo"),
                approval: ToolApproval::RequireConfirmation,
                max_calls_per_session: Some(10),
            }],
        };
        let json = serde_json::to_string(&policy).unwrap();
        let back: ToolPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.default_approval, ToolApproval::Auto);
        assert_eq!(back.rules.len(), 1);
        assert_eq!(back.rules[0].approval, ToolApproval::RequireConfirmation);
    }

    // -- validate_tool_input -----------------------------------------------

    #[test]
    fn validation_accepts_valid_input() {
        let desc = StubTool { id: "t", wire: "w" }.descriptor();
        let input = json!({"x": "hello"});
        assert!(validate_tool_input(&desc, &input).is_ok());
    }

    #[test]
    fn validation_rejects_non_object() {
        let desc = StubTool { id: "t", wire: "w" }.descriptor();
        let input = json!("not an object");
        let err = validate_tool_input(&desc, &input).unwrap_err();
        assert!(err.to_string().contains("must be a JSON object"));
    }

    #[test]
    fn validation_rejects_missing_required() {
        let desc = StubTool { id: "t", wire: "w" }.descriptor();
        let input = json!({});
        let err = validate_tool_input(&desc, &input).unwrap_err();
        assert!(err.to_string().contains("missing required field"));
        assert!(err.to_string().contains("x"));
    }

    #[test]
    fn validation_rejects_wrong_type() {
        let desc = StubTool { id: "t", wire: "w" }.descriptor();
        let input = json!({"x": 42});
        let err = validate_tool_input(&desc, &input).unwrap_err();
        assert!(err.to_string().contains("expected type \"string\""));
    }

    #[test]
    fn validation_allows_extra_fields() {
        let desc = StubTool { id: "t", wire: "w" }.descriptor();
        let input = json!({"x": "ok", "extra": true});
        assert!(validate_tool_input(&desc, &input).is_ok());
    }

    // -- EchoTool (behind test-utils feature) ------------------------------

    #[cfg(feature = "test-utils")]
    mod echo_tests {
        use super::*;
        use crate::test_utils::EchoTool;

        #[test]
        fn echo_descriptor() {
            let tool = EchoTool;
            let desc = tool.descriptor();
            assert_eq!(desc.wire_name, "echo");
            assert_eq!(desc.display_name, "Echo");
            assert_eq!(desc.id, ToolId::new("echo"));
            assert!(desc.parameters.get("properties").is_some());
        }

        #[tokio::test]
        async fn echo_execute() {
            let tool = EchoTool;
            let ctx = make_context();
            let result = tool.execute(json!({"message": "hello"}), &ctx).await.unwrap();
            assert_eq!(result, json!({"echoed": "hello"}));
        }

        #[tokio::test]
        async fn echo_missing_message_returns_empty() {
            let tool = EchoTool;
            let ctx = make_context();
            let result = tool.execute(json!({}), &ctx).await.unwrap();
            assert_eq!(result, json!({"echoed": ""}));
        }

        #[test]
        fn echo_validate_input() {
            let tool = EchoTool;
            let desc = tool.descriptor();
            assert!(validate_tool_input(&desc, &json!({"message": "hi"})).is_ok());
            assert!(validate_tool_input(&desc, &json!({})).is_err());
            assert!(validate_tool_input(&desc, &json!({"message": 123})).is_err());
        }
    }
}
