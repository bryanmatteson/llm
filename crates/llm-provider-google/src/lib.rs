pub mod auth;
pub mod client;
pub mod descriptor;
pub mod tools;
pub mod wire;

// ── Re-exports ──────────────────────────────────────────────────────

pub use auth::GoogleAuthProvider;
pub use client::GoogleClient;
pub use descriptor::{
    provider_descriptor, API_BASE, DEFAULT_MODEL, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
    PROVIDER_ID,
};
pub use tools::GoogleToolFormat;
