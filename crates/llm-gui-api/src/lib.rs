//! GUI-facing API layer for the LLM integration framework.
//!
//! This crate provides a thin, DTO-oriented facade that GUI frontends
//! (Tauri, Electron, web) can call without depending on internal framework
//! types. All public types are `Serialize` so they can be sent over any
//! transport.
//!
//! # Architecture
//!
//! ```text
//!   GUI frontend
//!       │
//!       ▼
//!   ┌───────────────┐
//!   │  GuiFacade     │  ← this crate
//!   └───┬───────────┘
//!       │  delegates to llm_app::AppContext services
//!       ▼
//!   ┌───────────────┐
//!   │  AppContext    │  ← llm_app (wired at startup via AppBuilder)
//!   └───────────────┘
//! ```

pub mod dto;
pub mod events;
pub mod facade;

// ── Re-exports ──────────────────────────────────────────────────────────

pub use dto::*;
pub use events::SessionEventAdapter;
pub use facade::GuiFacade;
