//! Terminal integration library for the LLM framework.
//!
//! This crate provides reusable building blocks for CLI applications that
//! integrate with the LLM framework:
//!
//! - [`approval::CliApprovalHandler`] — prompts the user for tool-call
//!   confirmation on stdin/stderr.
//! - [`render::questionnaire::run_terminal_questionnaire`] — drives a
//!   [`Questionnaire`](llm_questionnaire::Questionnaire) interactively in the
//!   terminal.
//! - [`render::stream::render_session_events`] — renders
//!   [`SessionEvent`](llm_session::SessionEvent)s to stderr as streaming text.

pub mod approval;
pub mod render;
