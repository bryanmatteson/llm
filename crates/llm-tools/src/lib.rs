pub mod context;
pub mod invocation;
pub mod policy;
pub mod registry;
pub mod tool;
pub mod validation;

pub use context::ToolContext;
pub use invocation::{ToolCall, ToolResult};
pub use policy::{ToolApproval, ToolPolicy, ToolPolicyBuilder, ToolPolicyRule};
pub use registry::ToolRegistry;
pub use tool::{DynTool, Tool, ToolDescriptor, ToolInfo};
pub use validation::validate_tool_input;

// Re-export schemars so users don't need to add it as a direct dependency.
pub use schemars;
pub use schemars::JsonSchema;

// ---------------------------------------------------------------------------
// Test-utilities feature: EchoTool
// ---------------------------------------------------------------------------

#[cfg(feature = "test-utils")]
pub mod test_utils {
    use async_trait::async_trait;
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};

    use llm_core::Result;

    use crate::context::ToolContext;
    use crate::tool::{Tool, ToolInfo};

    #[derive(Debug, Deserialize, JsonSchema)]
    pub struct EchoInput {
        /// The message to echo.
        pub message: String,
    }

    #[derive(Debug, Serialize)]
    pub struct EchoOutput {
        pub echoed: String,
    }

    /// A minimal tool that accepts `{"message": "<string>"}` and returns
    /// `{"echoed": "<string>"}`.
    #[derive(Debug)]
    pub struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        type Input = EchoInput;
        type Output = EchoOutput;

        fn info(&self) -> ToolInfo {
            ToolInfo::new("echo", "Echoes the input message back to the caller.")
                .display_name("Echo")
        }

        async fn execute(&self, input: EchoInput, _ctx: &ToolContext) -> Result<EchoOutput> {
            Ok(EchoOutput { echoed: input.message })
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
    use schemars::JsonSchema;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    use llm_core::{ModelId, ProviderId, Result, SessionId, ToolId};

    use crate::context::ToolContext;
    use crate::policy::{ToolApproval, ToolPolicy, ToolPolicyRule};
    use crate::registry::ToolRegistry;
    use crate::tool::{DynTool, Tool, ToolInfo};
    use crate::validation::validate_tool_input;

    // -- test tool -----------------------------------------------------------

    #[derive(Debug, Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct StubInput {
        x: String,
    }

    #[derive(Debug, Serialize)]
    struct StubOutput {
        ok: bool,
    }

    #[derive(Debug)]
    struct StubTool {
        id: &'static str,
        wire: &'static str,
    }

    #[async_trait]
    impl Tool for StubTool {
        type Input = StubInput;
        type Output = StubOutput;

        fn info(&self) -> ToolInfo {
            ToolInfo::new(self.id, "stub").wire_name(self.wire)
        }

        async fn execute(&self, _input: StubInput, _ctx: &ToolContext) -> Result<StubOutput> {
            Ok(StubOutput { ok: true })
        }
    }

    fn make_context() -> ToolContext {
        ToolContext {
            session_id: SessionId::new("test-session"),
            provider_id: ProviderId::new("test-provider"),
            model_id: ModelId::new("test-model"),
            metadata: BTreeMap::new(),
        }
    }

    // -- ToolRegistry --------------------------------------------------------

    #[test]
    fn registry_register_and_get() {
        let mut reg = ToolRegistry::new();
        let tool: Arc<dyn DynTool> = Arc::new(StubTool { id: "a", wire: "wire_a" });
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

    // -- ToolPolicy ----------------------------------------------------------

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
        assert_eq!(back, policy);
    }

    // -- ToolPolicyBuilder ---------------------------------------------------

    #[test]
    fn policy_builder_defaults_to_auto() {
        use crate::policy::ToolPolicyBuilder;
        let policy = ToolPolicyBuilder::new().build();
        assert_eq!(policy.default_approval, ToolApproval::Auto);
        assert!(policy.rules.is_empty());
    }

    #[test]
    fn policy_builder_fluent_chain() {
        use crate::policy::ToolPolicyBuilder;
        let policy = ToolPolicyBuilder::new()
            .default(ToolApproval::RequireConfirmation)
            .allow("search")
            .deny("dangerous")
            .confirm("file_write")
            .confirm_with("exec", 3)
            .allow_with("read", 10)
            .build();

        assert_eq!(policy.default_approval, ToolApproval::RequireConfirmation);
        assert_eq!(policy.rules.len(), 5);
        assert_eq!(policy.approval_for(&ToolId::new("search")), ToolApproval::Auto);
        assert_eq!(policy.approval_for(&ToolId::new("dangerous")), ToolApproval::Deny);
        assert_eq!(policy.approval_for(&ToolId::new("file_write")), ToolApproval::RequireConfirmation);
        assert_eq!(policy.approval_for(&ToolId::new("exec")), ToolApproval::RequireConfirmation);
        assert_eq!(policy.approval_for(&ToolId::new("unknown")), ToolApproval::RequireConfirmation);

        let exec_rule = policy.rules.iter().find(|r| r.tool_id == ToolId::new("exec")).unwrap();
        assert_eq!(exec_rule.max_calls_per_session, Some(3));

        let read_rule = policy.rules.iter().find(|r| r.tool_id == ToolId::new("read")).unwrap();
        assert_eq!(read_rule.max_calls_per_session, Some(10));
    }

    #[test]
    fn policy_builder_serde_roundtrip() {
        use crate::policy::ToolPolicyBuilder;
        let policy = ToolPolicyBuilder::new()
            .allow("a")
            .deny("b")
            .build();
        let json = serde_json::to_string(&policy).unwrap();
        let back: ToolPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, policy);
    }

    // -- Tool execution (via DynTool blanket) --------------------------------

    #[tokio::test]
    async fn tool_execute_with_valid_input() {
        let tool = StubTool { id: "t", wire: "w" };
        let ctx = make_context();
        let result = tool.invoke(json!({"x": "hello"}), &ctx).await.unwrap();
        assert_eq!(result, json!({"ok": true}));
    }

    #[tokio::test]
    async fn tool_rejects_invalid_input() {
        let tool = StubTool { id: "t", wire: "w" };
        let ctx = make_context();
        let err = tool.invoke(json!({"wrong": "field"}), &ctx).await;
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("deserialize"));
    }

    #[test]
    fn tool_descriptor_has_json_schema() {
        let tool = StubTool { id: "t", wire: "w" };
        let desc = tool.descriptor();
        let schema_str = serde_json::to_string(&desc.parameters).unwrap();
        assert!(schema_str.contains("\"x\""), "schema should contain the 'x' property");
    }

    // -- validate_tool_input -------------------------------------------------

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

    // -- EchoTool (behind test-utils feature) --------------------------------

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
        }

        #[tokio::test]
        async fn echo_execute() {
            let tool = EchoTool;
            let ctx = make_context();
            let result = tool.invoke(json!({"message": "hello"}), &ctx).await.unwrap();
            assert_eq!(result, json!({"echoed": "hello"}));
        }

        #[tokio::test]
        async fn echo_missing_message_returns_error() {
            let tool = EchoTool;
            let ctx = make_context();
            let err = tool.invoke(json!({}), &ctx).await;
            assert!(err.is_err());
        }
    }
}
