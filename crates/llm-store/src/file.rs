//! File-backed store implementations.
//!
//! Each store persists data as individual JSON files inside a configurable
//! directory.  The directory is created on first write if it does not already
//! exist.

use std::path::{Path, PathBuf};

use llm_auth::AuthSession;
use llm_core::{FrameworkError, ProviderId, Result, SessionId};

use crate::account::{AccountRecord, AccountStore};
use crate::credential::{CredentialStatus, CredentialStore};
use crate::session::{SessionSnapshot, SessionStore};

// ── helpers ─────────────────────────────────────────────────────────

/// Sanitize an id string so it is safe to use as a filename component.
/// Replaces any character that is not alphanumeric, `-`, or `_` with `_`.
fn safe_filename(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

fn ensure_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| {
        FrameworkError::storage(format!("failed to create directory {}: {e}", dir.display()))
    })
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|e| FrameworkError::storage(format!("serialization error: {e}")))?;
    std::fs::write(path, json).map_err(|e| {
        FrameworkError::storage(format!("failed to write {}: {e}", path.display()))
    })
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    match std::fs::read_to_string(path) {
        Ok(contents) => {
            let value = serde_json::from_str(&contents).map_err(|e| {
                FrameworkError::storage(format!(
                    "failed to parse {}: {e}",
                    path.display()
                ))
            })?;
            Ok(Some(value))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(FrameworkError::storage(format!(
            "failed to read {}: {e}",
            path.display()
        ))),
    }
}

fn remove_file(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(FrameworkError::storage(format!(
            "failed to remove {}: {e}",
            path.display()
        ))),
    }
}

/// List JSON file stems in `dir`, returning only those with the `.json` extension.
fn list_json_stems(dir: &Path) -> Result<Vec<String>> {
    match std::fs::read_dir(dir) {
        Ok(entries) => {
            let mut stems = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|e| {
                    FrameworkError::storage(format!("directory read error: {e}"))
                })?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        stems.push(stem.to_owned());
                    }
                }
            }
            Ok(stems)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(FrameworkError::storage(format!(
            "failed to read directory {}: {e}",
            dir.display()
        ))),
    }
}

// ── FileCredentialStore ─────────────────────────────────────────────

/// **TRANSITIONAL / DEV-ONLY** — stores credentials as plain-text JSON files.
///
/// This is intentionally insecure and exists only so that early development
/// can persist credentials without pulling in an OS-keychain dependency.
/// Production code should use an encrypted or keychain-backed implementation
/// of [`CredentialStore`].
#[derive(Debug, Clone)]
pub struct FileCredentialStore {
    dir: PathBuf,
}

impl FileCredentialStore {
    /// Create a new store rooted at `dir`.  The directory is created lazily
    /// on first write.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn api_key_path(&self, provider: &ProviderId) -> PathBuf {
        self.dir
            .join(format!("{}_apikey.json", safe_filename(provider.as_str())))
    }

    fn session_path(&self, provider: &ProviderId) -> PathBuf {
        self.dir
            .join(format!("{}_session.json", safe_filename(provider.as_str())))
    }
}

#[async_trait::async_trait]
impl CredentialStore for FileCredentialStore {
    async fn get_api_key(&self, provider: &ProviderId) -> Result<Option<String>> {
        read_json(&self.api_key_path(provider))
    }

    async fn set_api_key(&self, provider: &ProviderId, key: &str) -> Result<()> {
        ensure_dir(&self.dir)?;
        write_json(&self.api_key_path(provider), &key)
    }

    async fn clear_api_key(&self, provider: &ProviderId) -> Result<()> {
        remove_file(&self.api_key_path(provider))
    }

    async fn get_auth_session(&self, provider: &ProviderId) -> Result<Option<AuthSession>> {
        read_json(&self.session_path(provider))
    }

    async fn set_auth_session(&self, provider: &ProviderId, session: &AuthSession) -> Result<()> {
        ensure_dir(&self.dir)?;
        write_json(&self.session_path(provider), session)
    }

    async fn clear_auth_session(&self, provider: &ProviderId) -> Result<()> {
        remove_file(&self.session_path(provider))
    }

    async fn credential_status(&self, provider: &ProviderId) -> Result<CredentialStatus> {
        let has_api_key = std::fs::metadata(self.api_key_path(provider)).is_ok();
        let has_auth_session = std::fs::metadata(self.session_path(provider)).is_ok();
        Ok(CredentialStatus {
            has_api_key,
            has_auth_session,
        })
    }
}

// ── FileAccountStore ────────────────────────────────────────────────

/// Stores account records as individual `{provider_id}.json` files.
#[derive(Debug, Clone)]
pub struct FileAccountStore {
    dir: PathBuf,
}

impl FileAccountStore {
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn account_path(&self, provider: &ProviderId) -> PathBuf {
        self.dir
            .join(format!("{}.json", safe_filename(provider.as_str())))
    }
}

#[async_trait::async_trait]
impl AccountStore for FileAccountStore {
    async fn save_account(&self, account: &AccountRecord) -> Result<()> {
        ensure_dir(&self.dir)?;
        write_json(&self.account_path(&account.provider_id), account)
    }

    async fn list_accounts(&self) -> Result<Vec<AccountRecord>> {
        let stems = list_json_stems(&self.dir)?;
        let mut accounts = Vec::with_capacity(stems.len());
        for stem in &stems {
            let path = self.dir.join(format!("{stem}.json"));
            if let Some(record) = read_json::<AccountRecord>(&path)? {
                accounts.push(record);
            }
        }
        Ok(accounts)
    }

    async fn get_account(&self, provider: &ProviderId) -> Result<Option<AccountRecord>> {
        read_json(&self.account_path(provider))
    }

    async fn delete_account(&self, provider: &ProviderId) -> Result<()> {
        remove_file(&self.account_path(provider))
    }
}

// ── FileSessionStore ────────────────────────────────────────────────

/// Stores session snapshots as individual `{session_id}.json` files.
#[derive(Debug, Clone)]
pub struct FileSessionStore {
    dir: PathBuf,
}

impl FileSessionStore {
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn session_path(&self, id: &SessionId) -> PathBuf {
        self.dir
            .join(format!("{}.json", safe_filename(id.as_str())))
    }
}

#[async_trait::async_trait]
impl SessionStore for FileSessionStore {
    async fn save_session(&self, snapshot: &SessionSnapshot) -> Result<()> {
        ensure_dir(&self.dir)?;
        write_json(&self.session_path(&snapshot.id), snapshot)
    }

    async fn load_session(&self, id: &SessionId) -> Result<Option<SessionSnapshot>> {
        read_json(&self.session_path(id))
    }

    async fn list_sessions(&self) -> Result<Vec<SessionId>> {
        // Read each JSON file and extract the embedded `id` rather than
        // relying on the (possibly sanitized) filename stem.
        let stems = list_json_stems(&self.dir)?;
        let mut ids = Vec::with_capacity(stems.len());
        for stem in stems {
            let path = self.dir.join(format!("{stem}.json"));
            if let Some(snapshot) = read_json::<SessionSnapshot>(&path)? {
                ids.push(snapshot.id);
            }
        }
        Ok(ids)
    }

    async fn delete_session(&self, id: &SessionId) -> Result<()> {
        remove_file(&self.session_path(id))
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use llm_auth::{AuthMethod, TokenPair};
    use llm_core::{Message, Metadata, ModelId, ProviderId, SessionId};
    use tempfile::TempDir;

    use super::*;

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
            display_name: "File Account".into(),
            created_at: Utc::now(),
            metadata: Metadata::new(),
        }
    }

    fn test_session_snapshot() -> SessionSnapshot {
        SessionSnapshot {
            id: SessionId::new("sess-file-1"),
            provider_id: ProviderId::new("openai"),
            model: ModelId::new("gpt-4o"),
            system_prompt: None,
            metadata: Default::default(),
            messages: vec![Message::user("hello from file store")],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    // ── FileCredentialStore ─────────────────────────────────────────

    #[tokio::test]
    async fn file_credential_api_key_cycle() {
        let tmp = TempDir::new().unwrap();
        let store = FileCredentialStore::new(tmp.path().join("creds"));
        let pid = test_provider();

        assert!(store.get_api_key(&pid).await.unwrap().is_none());

        store.set_api_key(&pid, "sk-test-key").await.unwrap();
        assert_eq!(
            store.get_api_key(&pid).await.unwrap().as_deref(),
            Some("sk-test-key")
        );

        let status = store.credential_status(&pid).await.unwrap();
        assert!(status.has_api_key);
        assert!(!status.has_auth_session);

        store.clear_api_key(&pid).await.unwrap();
        assert!(store.get_api_key(&pid).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn file_credential_auth_session_cycle() {
        let tmp = TempDir::new().unwrap();
        let store = FileCredentialStore::new(tmp.path().join("creds"));
        let pid = test_provider();
        let session = test_auth_session(&pid);

        assert!(store.get_auth_session(&pid).await.unwrap().is_none());

        store.set_auth_session(&pid, &session).await.unwrap();
        let loaded = store.get_auth_session(&pid).await.unwrap().unwrap();
        assert_eq!(loaded.provider_id, pid);

        store.clear_auth_session(&pid).await.unwrap();
        assert!(store.get_auth_session(&pid).await.unwrap().is_none());
    }

    // ── FileAccountStore ────────────────────────────────────────────

    #[tokio::test]
    async fn file_account_write_read_delete() {
        let tmp = TempDir::new().unwrap();
        let store = FileAccountStore::new(tmp.path().join("accounts"));
        let pid = test_provider();
        let account = test_account(&pid);

        assert!(store.list_accounts().await.unwrap().is_empty());
        assert!(store.get_account(&pid).await.unwrap().is_none());

        store.save_account(&account).await.unwrap();
        let loaded = store.get_account(&pid).await.unwrap().unwrap();
        assert_eq!(loaded.display_name, "File Account");

        let all = store.list_accounts().await.unwrap();
        assert_eq!(all.len(), 1);

        store.delete_account(&pid).await.unwrap();
        assert!(store.get_account(&pid).await.unwrap().is_none());
        assert!(store.list_accounts().await.unwrap().is_empty());
    }

    // ── FileSessionStore ────────────────────────────────────────────

    #[tokio::test]
    async fn file_session_write_read_delete() {
        let tmp = TempDir::new().unwrap();
        let store = FileSessionStore::new(tmp.path().join("sessions"));
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
