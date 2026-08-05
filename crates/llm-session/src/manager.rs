use std::sync::Arc;

use async_trait::async_trait;

use chrono::{DateTime, Utc};
use llm_core::{FrameworkError, Result, SessionId, TokenUsage};
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
    /// Original creation time, preserved across snapshot updates.
    pub created_at: DateTime<Utc>,
}

/// Async trait for creating and retrieving sessions.
#[async_trait]
pub trait SessionManager: Send + Sync {
    /// Create a new session from the given configuration and return a handle.
    async fn create_session(&self, config: SessionConfig) -> Result<SessionHandle>;

    /// Retrieve a previously created session by its id.
    async fn get_session(&self, id: &SessionId) -> Result<Option<SessionHandle>>;

    /// Persist the complete canonical transcript, derived checkpoints, and
    /// cumulative usage for a live session.
    async fn save_session(&self, handle: &SessionHandle) -> Result<()>;

    /// List the ids of all known sessions.
    async fn list_sessions(&self) -> Result<Vec<SessionId>>;
}

/// Default [`SessionManager`] implementation backed by a [`SessionStore`].
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
        let created_at = Utc::now();
        let handle = SessionHandle {
            id,
            config,
            conversation: ConversationState::new(),
            total_usage: TokenUsage::default(),
            created_at,
        };
        let snapshot = SessionSnapshot {
            id: handle.id.clone(),
            config: handle.config.clone(),
            messages: vec![],
            checkpoints: vec![],
            total_usage: TokenUsage::default(),
            created_at,
            updated_at: created_at,
        };
        self.store.save_session(&snapshot).await?;
        Ok(handle)
    }

    async fn get_session(&self, id: &SessionId) -> Result<Option<SessionHandle>> {
        let snapshot = self.store.load_session(id).await?;
        match snapshot {
            Some(snap) => {
                let conversation = ConversationState::from_parts(snap.messages, snap.checkpoints);
                Ok(Some(SessionHandle {
                    id: snap.id,
                    config: snap.config,
                    conversation,
                    total_usage: snap.total_usage,
                    created_at: snap.created_at,
                }))
            }
            None => Ok(None),
        }
    }

    async fn save_session(&self, handle: &SessionHandle) -> Result<()> {
        self.store
            .save_session(&SessionSnapshot {
                id: handle.id.clone(),
                config: handle.config.clone(),
                messages: handle.conversation.messages().to_vec(),
                checkpoints: handle.conversation.checkpoints().to_vec(),
                total_usage: handle.total_usage.clone(),
                created_at: handle.created_at,
                updated_at: Utc::now(),
            })
            .await
    }

    async fn list_sessions(&self) -> Result<Vec<SessionId>> {
        self.store
            .list_sessions()
            .await
            .map_err(|e| FrameworkError::storage(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_core::ModelId;
    use llm_store::{ContextCheckpoint, ContextSummary, InMemorySessionStore};

    #[tokio::test]
    async fn save_restores_transcript_and_lifetime_usage() {
        let store = Arc::new(InMemorySessionStore::new());
        let manager = DefaultSessionManager::new(store);
        let mut handle = manager
            .create_session(SessionConfig::for_provider("test"))
            .await
            .unwrap();
        handle.conversation.append_user("persist me");
        handle.total_usage = TokenUsage {
            input_tokens: 12,
            output_tokens: 3,
        };
        handle.conversation.append_checkpoint(ContextCheckpoint {
            sequence: 1,
            covered_messages: 1,
            summary: ContextSummary {
                objective: Some("persist the session".into()),
                ..ContextSummary::default()
            },
            model: ModelId::new("summary-model"),
            estimated_tokens_before: 100,
            usage: TokenUsage {
                input_tokens: 8,
                output_tokens: 2,
            },
            created_at: Utc::now(),
        });

        manager.save_session(&handle).await.unwrap();
        let restored = manager.get_session(&handle.id).await.unwrap().unwrap();

        assert_eq!(restored.conversation.len(), 1);
        assert_eq!(
            restored.conversation.messages()[0].text_content(),
            "persist me"
        );
        assert_eq!(restored.total_usage.input_tokens, 12);
        assert_eq!(restored.total_usage.output_tokens, 3);
        assert_eq!(restored.conversation.checkpoints().len(), 1);
        assert_eq!(
            restored.conversation.checkpoints()[0]
                .summary
                .objective
                .as_deref(),
            Some("persist the session")
        );
        assert_eq!(restored.created_at, handle.created_at);
    }
}
