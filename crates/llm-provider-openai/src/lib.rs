pub mod auth;
pub mod client;
pub mod descriptor;
pub mod tools;
pub mod wire;

// ── Re-exports ──────────────────────────────────────────────────────

pub use auth::OpenAiAuthProvider;
pub use client::OpenAiClient;
pub use descriptor::{provider_descriptor, API_BASE, DEFAULT_MODEL, OPENAI_CLIENT_ID, PROVIDER_ID};
pub use tools::OpenAiToolFormat;
