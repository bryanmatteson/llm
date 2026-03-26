use chrono::{DateTime, Utc};
use llm_core::{Metadata, ProviderId, Result};
use serde::{Deserialize, Serialize};

/// A persisted account record linking a provider to a display name and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountRecord {
    pub provider_id: ProviderId,
    pub display_name: String,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: Metadata,
}

/// Async trait for CRUD operations on provider account records.
#[async_trait::async_trait]
pub trait AccountStore: Send + Sync {
    /// Insert or update an account record.
    async fn save_account(&self, account: &AccountRecord) -> Result<()>;

    /// List every stored account record.
    async fn list_accounts(&self) -> Result<Vec<AccountRecord>>;

    /// Retrieve the account record for `provider`, if any.
    async fn get_account(&self, provider: &ProviderId) -> Result<Option<AccountRecord>>;

    /// Delete the account record for `provider`.
    async fn delete_account(&self, provider: &ProviderId) -> Result<()>;
}
