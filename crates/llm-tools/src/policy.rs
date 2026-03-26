// ToolPolicy and related types now live in llm-core so they can be used by
// llm-store (via SessionConfig) without creating a circular dependency.
// Re-export everything for backward compatibility.
pub use llm_core::{ToolApproval, ToolPolicy, ToolPolicyBuilder, ToolPolicyRule};
