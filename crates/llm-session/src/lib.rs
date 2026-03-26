pub mod approval;
pub mod config;
pub mod conversation;
pub mod event;
pub mod limits;
pub mod manager;
pub mod mediator;

// ── Re-exports: core types ──────────────────────────────────────────

pub use approval::{ApprovalHandler, ApprovalRequest, ApprovalResponse, AutoApproveHandler};
pub use config::SessionConfig;
pub use conversation::ConversationState;
pub use event::{EventReceiver, EventSender, SessionEvent, event_channel};
pub use limits::SessionLimits;
pub use manager::{DefaultSessionManager, SessionHandle, SessionManager};
pub use mediator::{TurnOutcome, run_turn_loop};
