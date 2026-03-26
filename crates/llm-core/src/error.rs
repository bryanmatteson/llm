use crate::ids::{ProviderId, ToolId};

pub type Result<T> = std::result::Result<T, FrameworkError>;

#[derive(Debug, thiserror::Error)]
pub enum FrameworkError {
    #[error("authentication error: {reason}")]
    Auth { reason: String },

    #[error("provider error ({provider}): {message}")]
    Provider {
        provider: ProviderId,
        message: String,
    },

    #[error("tool execution error ({tool}): {reason}")]
    Tool { tool: ToolId, reason: String },

    #[error("session error: {reason}")]
    Session { reason: String },

    #[error("configuration error: {reason}")]
    Config { reason: String },

    #[error("questionnaire error: {reason}")]
    Questionnaire { reason: String },

    #[error("storage error: {reason}")]
    Storage { reason: String },

    #[error("validation error: {reason}")]
    Validation { reason: String },

    #[error("unsupported operation: {reason}")]
    Unsupported { reason: String },

    #[error("{0}")]
    Other(String),
}

impl FrameworkError {
    pub fn auth(reason: impl Into<String>) -> Self {
        Self::Auth {
            reason: reason.into(),
        }
    }

    pub fn provider(provider: ProviderId, message: impl Into<String>) -> Self {
        Self::Provider {
            provider,
            message: message.into(),
        }
    }

    pub fn tool(tool: ToolId, reason: impl Into<String>) -> Self {
        Self::Tool {
            tool,
            reason: reason.into(),
        }
    }

    pub fn session(reason: impl Into<String>) -> Self {
        Self::Session {
            reason: reason.into(),
        }
    }

    pub fn config(reason: impl Into<String>) -> Self {
        Self::Config {
            reason: reason.into(),
        }
    }

    pub fn questionnaire(reason: impl Into<String>) -> Self {
        Self::Questionnaire {
            reason: reason.into(),
        }
    }

    pub fn storage(reason: impl Into<String>) -> Self {
        Self::Storage {
            reason: reason.into(),
        }
    }

    pub fn validation(reason: impl Into<String>) -> Self {
        Self::Validation {
            reason: reason.into(),
        }
    }

    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self::Unsupported {
            reason: reason.into(),
        }
    }

    /// Returns `true` if the error is transient and the request may succeed
    /// on retry (e.g. HTTP 429 rate-limit or 503 service unavailable).
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Provider { message, .. } => {
                message.starts_with("HTTP 429") || message.starts_with("HTTP 503")
            }
            _ => false,
        }
    }
}
