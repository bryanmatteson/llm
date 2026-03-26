use llm_auth::AuthSession;
use llm_core::{ProviderId, Result};
use serde::{Deserialize, Serialize};

/// Summary of what credentials are available for a provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CredentialStatus {
    pub has_api_key: bool,
    pub has_auth_session: bool,
}

/// Async trait for storing and retrieving provider credentials.
///
/// Implementations range from in-memory (tests) through file-backed
/// (development) to OS-keychain or encrypted-at-rest (production).
#[async_trait::async_trait]
pub trait CredentialStore: Send + Sync {
    /// Retrieve the stored API key for `provider`, if any.
    async fn get_api_key(&self, provider: &ProviderId) -> Result<Option<String>>;

    /// Store an API key for `provider`, overwriting any previous value.
    async fn set_api_key(&self, provider: &ProviderId, key: &str) -> Result<()>;

    /// Remove the stored API key for `provider`.
    async fn clear_api_key(&self, provider: &ProviderId) -> Result<()>;

    /// Retrieve the stored [`AuthSession`] for `provider`, if any.
    async fn get_auth_session(&self, provider: &ProviderId) -> Result<Option<AuthSession>>;

    /// Persist an [`AuthSession`] for `provider`, overwriting any previous value.
    async fn set_auth_session(&self, provider: &ProviderId, session: &AuthSession) -> Result<()>;

    /// Remove the stored [`AuthSession`] for `provider`.
    async fn clear_auth_session(&self, provider: &ProviderId) -> Result<()>;

    /// Return a summary of what credentials are available for `provider`.
    async fn credential_status(&self, provider: &ProviderId) -> Result<CredentialStatus>;
}
