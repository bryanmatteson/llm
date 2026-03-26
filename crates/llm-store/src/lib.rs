pub mod account;
pub mod credential;
pub mod file;
pub mod memory;
pub mod session;

// ── Re-exports: traits ──────────────────────────────────────────────

pub use account::AccountStore;
pub use credential::CredentialStore;
pub use session::SessionStore;

// ── Re-exports: types ───────────────────────────────────────────────

pub use account::AccountRecord;
pub use credential::CredentialStatus;
pub use session::SessionSnapshot;

// ── Re-exports: in-memory implementations ───────────────────────────

pub use memory::{InMemoryAccountStore, InMemoryCredentialStore, InMemorySessionStore};

// ── Re-exports: file-backed implementations ─────────────────────────

pub use file::{FileAccountStore, FileCredentialStore, FileSessionStore};
