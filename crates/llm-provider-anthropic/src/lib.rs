pub mod auth;
pub mod client;
pub mod descriptor;
pub mod features;
pub mod tools;
pub mod wire;

// ── Re-exports ──────────────────────────────────────────────────────

pub use auth::AnthropicAuthProvider;
pub use client::AnthropicClient;
pub use descriptor::{
    ANTHROPIC_CLIENT_ID, API_BASE, DEFAULT_MODEL, PROVIDER_ID, provider_descriptor,
};
pub use features::{
    AnthropicMessageExt, AnthropicToolInfoExt, ClearToolUsesConfig,
    MESSAGE_CACHE_CONTROL_METADATA_KEY, TOOL_SEARCH_TOOL_BM25_20251119,
    TOOL_SEARCH_TOOL_REGEX_20251119, cache_control_ephemeral, cacheable_system_text_block,
    context_management, system_text_block, tool_reference_block, tool_search_bm25_tool,
    tool_search_regex_tool,
};
pub use tools::{
    AnthropicToolFormat, CODE_EXECUTION_20250825, CODE_EXECUTION_20260120, code_execution_tool,
};
