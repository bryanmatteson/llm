use serde::{Deserialize, Serialize};

use llm_core::ToolId;

/// Whether a tool invocation should be automatically approved, require explicit
/// human confirmation, or be outright denied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolApproval {
    /// The tool may be executed without further confirmation.
    Auto,
    /// A human must confirm each invocation before it proceeds.
    RequireConfirmation,
    /// The tool is not allowed to execute.
    Deny,
}

/// A per-tool policy rule that overrides the default approval setting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPolicyRule {
    /// The tool this rule applies to.
    pub tool_id: ToolId,
    /// The approval level for this tool.
    pub approval: ToolApproval,
    /// Optional cap on the number of times this tool may be called within a
    /// single session. `None` means unlimited.
    pub max_calls_per_session: Option<u32>,
}

/// Policy governing tool execution within a session.
///
/// The policy carries a default approval level and a set of per-tool rules
/// that override the default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPolicy {
    /// The fallback approval level used when no per-tool rule matches.
    pub default_approval: ToolApproval,
    /// Per-tool override rules.
    pub rules: Vec<ToolPolicyRule>,
}

impl ToolPolicy {
    /// Determine the [`ToolApproval`] for a given tool.
    ///
    /// If a matching rule exists the rule's approval is returned; otherwise the
    /// default approval is returned.
    pub fn approval_for(&self, tool_id: &ToolId) -> ToolApproval {
        self.rules
            .iter()
            .find(|r| r.tool_id == *tool_id)
            .map(|r| r.approval.clone())
            .unwrap_or_else(|| self.default_approval.clone())
    }

    /// Convenience: returns `true` when the tool is not [`ToolApproval::Deny`].
    pub fn is_allowed(&self, tool_id: &ToolId) -> bool {
        self.approval_for(tool_id) != ToolApproval::Deny
    }
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self {
            default_approval: ToolApproval::Auto,
            rules: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolPolicyBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a [`ToolPolicy`].
///
/// # Example
///
/// ```
/// use llm_tools::{ToolPolicyBuilder, ToolApproval};
///
/// let policy = ToolPolicyBuilder::new()
///     .default(ToolApproval::RequireConfirmation)
///     .allow("search")
///     .deny("dangerous_tool")
///     .confirm("file_write")
///     .confirm_with("exec", 3)
///     .build();
///
/// assert_eq!(policy.rules.len(), 4);
/// ```
pub struct ToolPolicyBuilder {
    default_approval: ToolApproval,
    rules: Vec<ToolPolicyRule>,
}

impl ToolPolicyBuilder {
    /// Create a new builder with `ToolApproval::Auto` as the default.
    pub fn new() -> Self {
        Self {
            default_approval: ToolApproval::Auto,
            rules: Vec::new(),
        }
    }

    /// Set the default approval level for tools without an explicit rule.
    pub fn default(mut self, approval: ToolApproval) -> Self {
        self.default_approval = approval;
        self
    }

    /// Allow a tool to execute without confirmation.
    pub fn allow(self, tool_id: impl Into<String>) -> Self {
        self.rule(tool_id, ToolApproval::Auto, None)
    }

    /// Deny a tool from executing.
    pub fn deny(self, tool_id: impl Into<String>) -> Self {
        self.rule(tool_id, ToolApproval::Deny, None)
    }

    /// Require human confirmation before executing a tool.
    pub fn confirm(self, tool_id: impl Into<String>) -> Self {
        self.rule(tool_id, ToolApproval::RequireConfirmation, None)
    }

    /// Require confirmation and limit the tool to `max_calls` per session.
    pub fn confirm_with(self, tool_id: impl Into<String>, max_calls: u32) -> Self {
        self.rule(tool_id, ToolApproval::RequireConfirmation, Some(max_calls))
    }

    /// Allow a tool with a per-session call limit.
    pub fn allow_with(self, tool_id: impl Into<String>, max_calls: u32) -> Self {
        self.rule(tool_id, ToolApproval::Auto, Some(max_calls))
    }

    /// Add a fully-specified rule.
    pub fn rule(
        mut self,
        tool_id: impl Into<String>,
        approval: ToolApproval,
        max_calls_per_session: Option<u32>,
    ) -> Self {
        self.rules.push(ToolPolicyRule {
            tool_id: ToolId::new(tool_id.into()),
            approval,
            max_calls_per_session,
        });
        self
    }

    /// Build the [`ToolPolicy`].
    pub fn build(self) -> ToolPolicy {
        ToolPolicy {
            default_approval: self.default_approval,
            rules: self.rules,
        }
    }
}

impl Default for ToolPolicyBuilder {
    fn default() -> Self {
        Self::new()
    }
}
