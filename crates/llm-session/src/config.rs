// SessionConfig now lives in llm-core so it can be used by llm-store
// (in SessionSnapshot) without circular dependencies. Re-export here.
pub use llm_core::SessionConfig;
