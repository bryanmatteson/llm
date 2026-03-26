pub mod auth;
pub mod client;
pub mod descriptor;
pub mod tools;
pub mod wire;

// ── Re-exports ──────────────────────────────────────────────────────

pub use auth::AnthropicAuthProvider;
pub use client::AnthropicClient;
pub use descriptor::{provider_descriptor, API_BASE, ANTHROPIC_CLIENT_ID, DEFAULT_MODEL, PROVIDER_ID};
pub use tools::AnthropicToolFormat;
