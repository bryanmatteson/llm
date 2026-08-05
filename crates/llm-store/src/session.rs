use chrono::{DateTime, Utc};
use llm_core::{Message, ModelId, Result, SessionConfig, SessionId, TokenUsage};
use serde::{Deserialize, Serialize};

/// Structured memory for a compacted transcript prefix. The exact messages
/// remain in [`SessionSnapshot::messages`]; this is only a provider-context
/// projection.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextSummary {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decisions: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub open_items: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifacts: Vec<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub narrative: String,
}

/// An auditable checkpoint covering a prefix of the canonical transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextCheckpoint {
    pub sequence: u64,
    /// Exclusive message index in [`SessionSnapshot::messages`].
    pub covered_messages: usize,
    pub summary: ContextSummary,
    pub model: ModelId,
    pub estimated_tokens_before: u64,
    #[serde(default)]
    pub usage: TokenUsage,
    pub created_at: DateTime<Utc>,
}

/// A point-in-time snapshot of a conversation session, suitable for
/// persistence and later restoration.
///
/// Stores the full [`SessionConfig`] so that `get_session` restores the
/// exact configuration (tool policy, limits, system prompt, etc.) that
/// was active at creation time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub id: SessionId,
    pub config: SessionConfig,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoints: Vec<ContextCheckpoint>,
    #[serde(default)]
    pub total_usage: TokenUsage,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_from_before_compaction_remain_readable() {
        let snapshot: SessionSnapshot = serde_json::from_value(serde_json::json!({
            "id": "legacy-session",
            "config": {
                "provider_id": "test",
                "model": null,
                "system_prompt": null
            },
            "messages": [],
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z"
        }))
        .unwrap();

        assert!(snapshot.checkpoints.is_empty());
        assert_eq!(snapshot.total_usage.total(), 0);
    }
}
