use std::collections::HashMap;

use llm_auth::AuthSession;
use llm_core::{ProviderId, Result, SessionId};
use tokio::sync::RwLock;

use crate::account::{AccountRecord, AccountStore};
use crate::credential::{CredentialStatus, CredentialStore};
use crate::session::{SessionSnapshot, SessionStore};

// ── InMemoryCredentialStore ─────────────────────────────────────────

/// Credentials held in memory — suitable for tests and short-lived processes.
#[derive(Debug, Default)]
pub struct InMemoryCredentialStore {
    api_keys: RwLock<HashMap<ProviderId, String>>,
    sessions: RwLock<HashMap<ProviderId, AuthSession>>,
}

impl InMemoryCredentialStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl CredentialStore for InMemoryCredentialStore {
    async fn get_api_key(&self, provider: &ProviderId) -> Result<Option<String>> {
        Ok(self.api_keys.read().await.get(provider).cloned())
    }

    async fn set_api_key(&self, provider: &ProviderId, key: &str) -> Result<()> {
        self.api_keys
            .write()
            .await
            .insert(provider.clone(), key.to_owned());
        Ok(())
    }

    async fn clear_api_key(&self, provider: &ProviderId) -> Result<()> {
        self.api_keys.write().await.remove(provider);
        Ok(())
    }

    async fn get_auth_session(&self, provider: &ProviderId) -> Result<Option<AuthSession>> {
        Ok(self.sessions.read().await.get(provider).cloned())
    }

    async fn set_auth_session(&self, provider: &ProviderId, session: &AuthSession) -> Result<()> {
        self.sessions
            .write()
            .await
            .insert(provider.clone(), session.clone());
        Ok(())
    }

    async fn clear_auth_session(&self, provider: &ProviderId) -> Result<()> {
        self.sessions.write().await.remove(provider);
        Ok(())
    }

    async fn credential_status(&self, provider: &ProviderId) -> Result<CredentialStatus> {
        let has_api_key = self.api_keys.read().await.contains_key(provider);
        let has_auth_session = self.sessions.read().await.contains_key(provider);
        Ok(CredentialStatus {
            has_api_key,
            has_auth_session,
        })
    }
}

// ── InMemoryAccountStore ────────────────────────────────────────────

/// Account records held in memory.
#[derive(Debug, Default)]
pub struct InMemoryAccountStore {
    accounts: RwLock<HashMap<ProviderId, AccountRecord>>,
}

impl InMemoryAccountStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl AccountStore for InMemoryAccountStore {
    async fn save_account(&self, account: &AccountRecord) -> Result<()> {
        self.accounts
            .write()
            .await
            .insert(account.provider_id.clone(), account.clone());
        Ok(())
    }

    async fn list_accounts(&self) -> Result<Vec<AccountRecord>> {
        Ok(self.accounts.read().await.values().cloned().collect())
    }

    async fn get_account(&self, provider: &ProviderId) -> Result<Option<AccountRecord>> {
        Ok(self.accounts.read().await.get(provider).cloned())
    }

    async fn delete_account(&self, provider: &ProviderId) -> Result<()> {
        self.accounts.write().await.remove(provider);
        Ok(())
    }
}

// ── InMemorySessionStore ────────────────────────────────────────────

/// Session snapshots held in memory.
#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    sessions: RwLock<HashMap<SessionId, SessionSnapshot>>,
}

impl InMemorySessionStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait::async_trait]
impl SessionStore for InMemorySessionStore {
    async fn save_session(&self, snapshot: &SessionSnapshot) -> Result<()> {
        self.sessions
            .write()
            .await
            .insert(snapshot.id.clone(), snapshot.clone());
        Ok(())
    }

    async fn load_session(&self, id: &SessionId) -> Result<Option<SessionSnapshot>> {
        Ok(self.sessions.read().await.get(id).cloned())
    }

    async fn list_sessions(&self) -> Result<Vec<SessionId>> {
        Ok(self.sessions.read().await.keys().cloned().collect())
    }

    async fn delete_session(&self, id: &SessionId) -> Result<()> {
        self.sessions.write().await.remove(id);
        Ok(())
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use llm_auth::{AuthMethod, TokenPair};
    use llm_core::{Message, Metadata, ModelId, ProviderId, SessionId};

    use super::*;

    // ── helpers ──────────────────────────────────────────────────────

    fn test_provider() -> ProviderId {
        ProviderId::new("test-provider")
    }

    fn test_auth_session(provider: &ProviderId) -> AuthSession {
        AuthSession {
            provider_id: provider.clone(),
            method: AuthMethod::ApiKey {
                masked: "sk-****1234".into(),
            },
            tokens: TokenPair::new("access-tok".into(), Some("refresh-tok".into()), 3600),
            metadata: Metadata::new(),
        }
    }

    fn test_account(provider: &ProviderId) -> AccountRecord {
        AccountRecord {
            provider_id: provider.clone(),
            display_name: "Test Account".into(),
            created_at: Utc::now(),
            metadata: Metadata::new(),
        }
    }

    fn test_session_snapshot() -> SessionSnapshot {
        use llm_core::SessionConfig;
        SessionSnapshot {
            id: SessionId::new("sess-1"),
            config: SessionConfig {
                provider_id: ProviderId::new("openai"),
                model: Some(ModelId::new("gpt-4o")),
                system_prompt: None,
                tool_policy: Default::default(),
                limits: Default::default(),
                metadata: Default::default(),
                provider_tools: Vec::new(),
                provider_request: Default::default(),
            },
            messages: vec![Message::user("hello")],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    // ── credential store ────────────────────────────────────────────

    #[tokio::test]
    async fn credential_api_key_crud() {
        let store = InMemoryCredentialStore::new();
        let pid = test_provider();

        // Initially empty.
        assert!(store.get_api_key(&pid).await.unwrap().is_none());

        // Set and read back.
        store.set_api_key(&pid, "sk-secret").await.unwrap();
        assert_eq!(
            store.get_api_key(&pid).await.unwrap().as_deref(),
            Some("sk-secret")
        );

        // Status reflects key.
        let status = store.credential_status(&pid).await.unwrap();
        assert!(status.has_api_key);
        assert!(!status.has_auth_session);

        // Clear.
        store.clear_api_key(&pid).await.unwrap();
        assert!(store.get_api_key(&pid).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn credential_auth_session_crud() {
        let store = InMemoryCredentialStore::new();
        let pid = test_provider();
        let session = test_auth_session(&pid);

        assert!(store.get_auth_session(&pid).await.unwrap().is_none());

        store.set_auth_session(&pid, &session).await.unwrap();
        let loaded = store.get_auth_session(&pid).await.unwrap().unwrap();
        assert_eq!(loaded.provider_id, pid);

        let status = store.credential_status(&pid).await.unwrap();
        assert!(status.has_auth_session);

        store.clear_auth_session(&pid).await.unwrap();
        assert!(store.get_auth_session(&pid).await.unwrap().is_none());
    }

    // ── account store ───────────────────────────────────────────────

    #[tokio::test]
    async fn account_crud() {
        let store = InMemoryAccountStore::new();
        let pid = test_provider();
        let account = test_account(&pid);

        // Empty initially.
        assert!(store.list_accounts().await.unwrap().is_empty());
        assert!(store.get_account(&pid).await.unwrap().is_none());

        // Save and retrieve.
        store.save_account(&account).await.unwrap();
        let loaded = store.get_account(&pid).await.unwrap().unwrap();
        assert_eq!(loaded.display_name, "Test Account");

        // List.
        let all = store.list_accounts().await.unwrap();
        assert_eq!(all.len(), 1);

        // Delete.
        store.delete_account(&pid).await.unwrap();
        assert!(store.get_account(&pid).await.unwrap().is_none());
        assert!(store.list_accounts().await.unwrap().is_empty());
    }

    // ── session store ───────────────────────────────────────────────

    #[tokio::test]
    async fn session_crud() {
        let store = InMemorySessionStore::new();
        let snapshot = test_session_snapshot();
        let sid = snapshot.id.clone();

        assert!(store.list_sessions().await.unwrap().is_empty());
        assert!(store.load_session(&sid).await.unwrap().is_none());

        store.save_session(&snapshot).await.unwrap();
        let loaded = store.load_session(&sid).await.unwrap().unwrap();
        assert_eq!(loaded.id, sid);
        assert_eq!(loaded.messages.len(), 1);

        let ids = store.list_sessions().await.unwrap();
        assert_eq!(ids.len(), 1);

        store.delete_session(&sid).await.unwrap();
        assert!(store.load_session(&sid).await.unwrap().is_none());
        assert!(store.list_sessions().await.unwrap().is_empty());
    }
}
