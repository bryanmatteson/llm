//! Fluent builder for [`SessionConfig`].
//!
//! # Example
//!
//! ```
//! use llm_session::SessionBuilder;
//! use llm_tools::ToolApproval;
//!
//! let config = SessionBuilder::new("openai")
//!     .model("gpt-4o")
//!     .system_prompt("You are a helpful assistant.")
//!     .max_turns(20)
//!     .turn_timeout_secs(180)
//!     .tool_timeout_secs(60)
//!     .max_tool_calls_per_turn(12)
//!     .default_tool_approval(ToolApproval::Auto)
//!     .deny_tool("exec_shell")
//!     .confirm_tool("delete_file")
//!     .confirm_tool_with_limit("web_search", 10)
//!     .build();
//! ```

use std::time::Duration;

use llm_core::{Metadata, ModelId, ProviderId, ToolId};
use llm_tools::{ToolApproval, ToolPolicy, ToolPolicyRule};

use crate::config::SessionConfig;
use crate::limits::SessionLimits;

/// Fluent builder for a [`SessionConfig`].
pub struct SessionBuilder {
    provider_id: ProviderId,
    model: Option<ModelId>,
    system_prompt: Option<String>,
    default_approval: ToolApproval,
    rules: Vec<ToolPolicyRule>,
    limits: SessionLimits,
    metadata: Metadata,
}

impl SessionBuilder {
    /// Start building a session config for the given provider.
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider_id: ProviderId::new(provider.into()),
            model: None,
            system_prompt: None,
            default_approval: ToolApproval::Auto,
            rules: Vec::new(),
            limits: SessionLimits::default(),
            metadata: Metadata::new(),
        }
    }

    /// Set the model to use.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(ModelId::new(model.into()));
        self
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    // -- Limits ---------------------------------------------------------------

    /// Set the maximum number of LLM turns.
    pub fn max_turns(mut self, n: usize) -> Self {
        self.limits.max_turns = n;
        self
    }

    /// Set the maximum tool calls allowed per turn.
    pub fn max_tool_calls_per_turn(mut self, n: usize) -> Self {
        self.limits.max_tool_calls_per_turn = n;
        self
    }

    /// Set the turn timeout in seconds.
    pub fn turn_timeout_secs(mut self, secs: u64) -> Self {
        self.limits.turn_timeout = Duration::from_secs(secs);
        self
    }

    /// Set the tool execution timeout in seconds.
    pub fn tool_timeout_secs(mut self, secs: u64) -> Self {
        self.limits.tool_timeout = Duration::from_secs(secs);
        self
    }

    /// Set the full `SessionLimits` directly.
    pub fn limits(mut self, limits: SessionLimits) -> Self {
        self.limits = limits;
        self
    }

    // -- Tool policy ----------------------------------------------------------

    /// Set the default tool approval level.
    pub fn default_tool_approval(mut self, approval: ToolApproval) -> Self {
        self.default_approval = approval;
        self
    }

    /// Add a tool that requires human confirmation before execution.
    pub fn confirm_tool(mut self, tool: impl Into<String>) -> Self {
        self.rules.push(ToolPolicyRule {
            tool_id: ToolId::new(tool.into()),
            approval: ToolApproval::RequireConfirmation,
            max_calls_per_session: None,
        });
        self
    }

    /// Add a tool that requires confirmation with a per-session call limit.
    pub fn confirm_tool_with_limit(mut self, tool: impl Into<String>, max: u32) -> Self {
        self.rules.push(ToolPolicyRule {
            tool_id: ToolId::new(tool.into()),
            approval: ToolApproval::RequireConfirmation,
            max_calls_per_session: Some(max),
        });
        self
    }

    /// Deny a tool entirely.
    pub fn deny_tool(mut self, tool: impl Into<String>) -> Self {
        self.rules.push(ToolPolicyRule {
            tool_id: ToolId::new(tool.into()),
            approval: ToolApproval::Deny,
            max_calls_per_session: None,
        });
        self
    }

    /// Auto-approve a tool (useful when the default is `RequireConfirmation`).
    pub fn auto_tool(mut self, tool: impl Into<String>) -> Self {
        self.rules.push(ToolPolicyRule {
            tool_id: ToolId::new(tool.into()),
            approval: ToolApproval::Auto,
            max_calls_per_session: None,
        });
        self
    }

    /// Set the full tool policy directly (replaces default_approval and rules).
    pub fn tool_policy(mut self, policy: ToolPolicy) -> Self {
        self.default_approval = policy.default_approval;
        self.rules = policy.rules;
        self
    }

    // -- Metadata -------------------------------------------------------------

    /// Add a metadata key-value pair.
    pub fn meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    // -- Build ----------------------------------------------------------------

    /// Consume the builder and produce a [`SessionConfig`].
    pub fn build(self) -> SessionConfig {
        SessionConfig {
            provider_id: self.provider_id,
            model: self.model,
            system_prompt: self.system_prompt,
            tool_policy: ToolPolicy {
                default_approval: self.default_approval,
                rules: self.rules,
            },
            limits: self.limits,
            metadata: self.metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_builder() {
        let config = SessionBuilder::new("openai").build();
        assert_eq!(config.provider_id.as_str(), "openai");
        assert!(config.model.is_none());
        assert!(config.system_prompt.is_none());
        assert_eq!(config.limits.max_turns, 10); // default
    }

    #[test]
    fn full_builder() {
        let config = SessionBuilder::new("anthropic")
            .model("claude-4")
            .system_prompt("Be concise.")
            .max_turns(30)
            .turn_timeout_secs(300)
            .tool_timeout_secs(60)
            .max_tool_calls_per_turn(4)
            .default_tool_approval(ToolApproval::RequireConfirmation)
            .auto_tool("search")
            .deny_tool("exec_shell")
            .confirm_tool_with_limit("delete_file", 3)
            .meta("user", "alice")
            .build();

        assert_eq!(config.provider_id.as_str(), "anthropic");
        assert_eq!(config.model.as_ref().unwrap().as_str(), "claude-4");
        assert_eq!(config.system_prompt.as_deref(), Some("Be concise."));
        assert_eq!(config.limits.max_turns, 30);
        assert_eq!(config.limits.turn_timeout, Duration::from_secs(300));
        assert_eq!(config.limits.tool_timeout, Duration::from_secs(60));
        assert_eq!(config.limits.max_tool_calls_per_turn, 4);
        assert_eq!(
            config.tool_policy.default_approval,
            ToolApproval::RequireConfirmation
        );
        assert_eq!(config.tool_policy.rules.len(), 3);
        assert_eq!(
            config.tool_policy.approval_for(&ToolId::new("search")),
            ToolApproval::Auto
        );
        assert_eq!(
            config.tool_policy.approval_for(&ToolId::new("exec_shell")),
            ToolApproval::Deny
        );
        assert_eq!(
            config.tool_policy.approval_for(&ToolId::new("delete_file")),
            ToolApproval::RequireConfirmation
        );
        assert_eq!(config.metadata.get("user").unwrap(), "alice");
    }

    #[test]
    fn tool_policy_override() {
        let policy = ToolPolicy {
            default_approval: ToolApproval::Deny,
            rules: vec![ToolPolicyRule {
                tool_id: ToolId::new("safe"),
                approval: ToolApproval::Auto,
                max_calls_per_session: None,
            }],
        };
        let config = SessionBuilder::new("openai").tool_policy(policy).build();

        assert_eq!(config.tool_policy.default_approval, ToolApproval::Deny);
        assert_eq!(config.tool_policy.rules.len(), 1);
    }
}
