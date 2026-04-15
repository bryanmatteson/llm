pub mod auth;
pub mod client;
pub mod descriptor;
pub mod tools;
pub mod wire;

// ── Re-exports ──────────────────────────────────────────────────────

pub use auth::OpenAiAuthProvider;
pub use client::OpenAiClient;
pub use descriptor::{
    API_BASE, CHATGPT_API_BASE, DEFAULT_MODEL, OPENAI_CLIENT_ID, PROVIDER_ID, provider_descriptor,
};
pub use tools::OpenAiToolFormat;
