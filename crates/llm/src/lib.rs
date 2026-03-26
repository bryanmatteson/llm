//! # llm
//!
//! Unified facade for the LLM framework.  Instead of depending on 12 crates
//! individually, add a single dependency:
//!
//! ```toml
//! [dependencies]
//! llm = { git = "https://github.com/bryanmatteson/llm.git" }
//! ```
//!
//! Everything is re-exported under feature-gated modules.  The `default`
//! feature enables all providers and both CLI/GUI adapters.  Disable what
//! you don't need:
//!
//! ```toml
//! llm = { git = "...", default-features = false, features = ["openai", "cli"] }
//! ```

// ── Core (always available) ────────────────────────────────────────

/// Foundational types: IDs, errors, messages, config, policy.
pub use llm_core as core;

/// Provider authentication: OAuth, API keys, token management.
pub use llm_auth as auth;

/// Typed tool system: define tools, registries, policies.
pub use llm_tools as tools;

/// Data-driven questionnaire engine.
pub use llm_questionnaire as questionnaire;

/// TOML configuration loading.
pub use llm_config as config;

/// Pluggable persistence: credential, session, and account stores.
pub use llm_store as store;

/// Provider client trait and wire types.
pub use llm_provider_api as provider_api;

/// Session orchestration: turn loops, streaming, approval, events.
pub use llm_session as session;

/// Application wiring: builder, services, provider registry.
pub use llm_app as app;

// ── Providers (feature-gated) ──────────────────────────────────────

/// OpenAI / GPT provider.
#[cfg(feature = "openai")]
pub use llm_provider_openai as openai;

/// Anthropic / Claude provider.
#[cfg(feature = "anthropic")]
pub use llm_provider_anthropic as anthropic;

/// Google / Gemini provider.
#[cfg(feature = "google")]
pub use llm_provider_google as google;

// ── MCP server (feature-gated) ────────────────────────────────────

/// Generic MCP server skeleton: protocol handler, tool dispatch, transports.
#[cfg(feature = "mcp")]
pub use llm_mcp as mcp;

// ── Frontend adapters (feature-gated) ──────────────────────────────

/// Terminal utilities: questionnaire renderer, stream renderer, approval handler.
#[cfg(feature = "cli")]
pub use llm_cli as cli;

/// GUI facade: DTOs, event adapter, async service layer.
#[cfg(feature = "gui")]
pub use llm_gui_api as gui;

// ── Convenience re-exports ─────────────────────────────────────────
//
// The most commonly used types, available at `llm::` without digging
// into submodules.

pub use llm_core::{
    FrameworkError, Message, Metadata, ModelId, ProviderId, Result, SessionConfig, SessionId,
    SessionLimits, StopReason, TokenUsage, ToolApproval, ToolId, ToolPolicy, ToolPolicyBuilder,
};

pub use llm_app::{AppBuilder, LlmContext};
pub use llm_auth::{
    AuthSession, CredentialKind, EnvCredentialDiscovery, ProviderCredential,
    build_auth_session as build_auth_session_from_credential,
};
pub use llm_provider_api::{LlmProviderClient, RetryConfig, RetryingClient};
pub use llm_questionnaire::{AnswerMap, AnswerValue, QuestionnaireBuilder, QuestionnaireRun};
pub use llm_session::{
    ConversationState, EventReceiver, EventSender, SessionEvent, SessionHandle, TurnLoopContext,
    TurnOutcome,
};
pub use llm_store::{CredentialStore, SessionStore};
pub use llm_tools::{DynTool, FnTool, Tool, ToolDescriptor, ToolInfo, ToolRegistry};
