pub mod auth;
pub mod client;
pub mod descriptor;
pub mod tools;
pub mod wire;

// ── Re-exports ──────────────────────────────────────────────────────

pub use auth::AnthropicAuthProvider;
pub use client::AnthropicClient;
pub use descriptor::{
    ANTHROPIC_CLIENT_ID, API_BASE, DEFAULT_MODEL, PROVIDER_ID, provider_descriptor,
};
pub use tools::AnthropicToolFormat;
