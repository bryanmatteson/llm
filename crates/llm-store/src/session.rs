use chrono::{DateTime, Utc};
use llm_core::{Message, Metadata, ModelId, ProviderId, Result, SessionId};
use serde::{Deserialize, Serialize};

/// A point-in-time snapshot of a conversation session, suitable for
/// persistence and later restoration.
///
/// Captures the full session configuration so that `get_session` can
/// restore the exact same config that was used at creation time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub id: SessionId,
    pub provider_id: ProviderId,
    pub model: ModelId,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub metadata: Metadata,
    pub messages: Vec<Message>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Async trait for persisting and loading conversation session snapshots.
#[async_trait::async_trait]
pub trait SessionStore: Send + Sync {
    /// Insert or update a session snapshot.
    async fn save_session(&self, snapshot: &SessionSnapshot) -> Result<()>;

    /// Load a previously-saved session snapshot by its id.
    async fn load_session(&self, id: &SessionId) -> Result<Option<SessionSnapshot>>;

    /// List the ids of all stored sessions.
    async fn list_sessions(&self) -> Result<Vec<SessionId>>;

    /// Delete a session snapshot by its id.
    async fn delete_session(&self, id: &SessionId) -> Result<()>;
}
