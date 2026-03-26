use std::sync::Arc;

use async_trait::async_trait;

use chrono::Utc;
use llm_core::{FrameworkError, ModelId, Result, SessionId, TokenUsage};
use llm_store::{SessionSnapshot, SessionStore};

use crate::config::SessionConfig;
use crate::conversation::ConversationState;

/// A live session handle bundling its identity, configuration, and
/// conversation state.
#[derive(Debug, Clone)]
pub struct SessionHandle {
    /// Unique identifier for this session.
    pub id: SessionId,
    /// The configuration that governs this session's turn loop.
    pub config: SessionConfig,
    /// The mutable conversation transcript.
    pub conversation: ConversationState,
    /// Cumulative token usage across all `send_message` calls for this session.
    pub total_usage: TokenUsage,
}

/// Async trait for creating and retrieving sessions.
#[async_trait]
pub trait SessionManager: Send + Sync {
    /// Create a new session from the given configuration and return a handle.
    async fn create_session(&self, config: SessionConfig) -> Result<SessionHandle>;

    /// Retrieve a previously created session by its id.
    async fn get_session(&self, id: &SessionId) -> Result<Option<SessionHandle>>;

    /// List the ids of all known sessions.
    async fn list_sessions(&self) -> Result<Vec<SessionId>>;
}

/// Default [`SessionManager`] implementation backed by a [`SessionStore`].
///
/// Session IDs are generated as random UUIDs (v4-style, using the `fastrand`
/// crate to avoid a dependency on `uuid`).
pub struct DefaultSessionManager {
    store: Arc<dyn SessionStore>,
}

impl std::fmt::Debug for DefaultSessionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DefaultSessionManager")
            .field("store", &"<dyn SessionStore>")
            .finish()
    }
}

impl DefaultSessionManager {
    pub fn new(store: Arc<dyn SessionStore>) -> Self {
        Self { store }
    }

    /// Generate a pseudo-random UUID-v4-style session id.
    fn generate_id() -> SessionId {
        let a: u64 = fastrand::u64(..);
        let b: u64 = fastrand::u64(..);
        SessionId::new(format!("{a:016x}-{b:016x}"))
    }
}

#[async_trait]
impl SessionManager for DefaultSessionManager {
    async fn create_session(&self, config: SessionConfig) -> Result<SessionHandle> {
        let id = Self::generate_id();
        let handle = SessionHandle {
            id,
            config,
            conversation: ConversationState::new(),
            total_usage: TokenUsage::default(),
        };
        let snapshot = SessionSnapshot {
            id: handle.id.clone(),
            provider_id: handle.config.provider_id.clone(),
            model: handle.config.model.clone().unwrap_or_else(|| ModelId::new("unknown")),
            system_prompt: handle.config.system_prompt.clone(),
            metadata: handle.config.metadata.clone(),
            messages: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        self.store.save_session(&snapshot).await?;
        Ok(handle)
    }

    async fn get_session(&self, id: &SessionId) -> Result<Option<SessionHandle>> {
        let snapshot = self.store.load_session(id).await?;
        match snapshot {
            Some(snap) => {
                // Reconstruct a SessionHandle from the persisted snapshot.
                // We build a minimal config from what the snapshot recorded.
                let config = SessionConfig {
                    provider_id: snap.provider_id,
                    model: Some(snap.model),
                    system_prompt: snap.system_prompt,
                    tool_policy: Default::default(),
                    limits: Default::default(),
                    metadata: snap.metadata,
                };
                let mut conversation = ConversationState::new();
                for msg in snap.messages {
                    conversation.append_message(msg);
                }
                Ok(Some(SessionHandle {
                    id: snap.id,
                    config,
                    conversation,
                    total_usage: TokenUsage::default(),
                }))
            }
            None => Ok(None),
        }
    }

    async fn list_sessions(&self) -> Result<Vec<SessionId>> {
        self.store
            .list_sessions()
            .await
            .map_err(|e| FrameworkError::storage(e.to_string()))
    }
}
