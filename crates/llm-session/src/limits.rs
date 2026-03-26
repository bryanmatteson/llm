// SessionLimits now lives in llm-core so it can be used by llm-store
// without creating a circular dependency. Re-export for backward compat.
pub use llm_core::SessionLimits;
