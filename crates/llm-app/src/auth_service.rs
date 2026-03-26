use std::sync::Arc;

use chrono::Utc;

use llm_auth::{AuthCompletion, AuthSession, AuthStart};
use llm_core::{FrameworkError, Metadata, ProviderId, Result};
use llm_store::{AccountRecord, AccountStore, CredentialStatus, CredentialStore};

use crate::registry::ProviderRegistry;

/// High-level authentication service that coordinates provider auth flows
/// with the credential and account stores.
pub struct AuthService {
    provider_registry: Arc<ProviderRegistry>,
    credential_store: Arc<dyn CredentialStore>,
    account_store: Arc<dyn AccountStore>,
}

impl AuthService {
    /// Create a new `AuthService`.
    pub fn new(
        provider_registry: Arc<ProviderRegistry>,
        credential_store: Arc<dyn CredentialStore>,
        account_store: Arc<dyn AccountStore>,
    ) -> Self {
        Self {
            provider_registry,
            credential_store,
            account_store,
        }
    }

    /// Begin a login flow for the given provider.
    ///
    /// Returns an [`AuthStart`] that tells the presentation layer what to
    /// show the user (browser URL, device code, API-key prompt, etc.).
    pub async fn start_login(&self, provider_id: &ProviderId) -> Result<AuthStart> {
        let registration = self
            .provider_registry
            .get(provider_id)
            .ok_or_else(|| FrameworkError::auth(format!("unknown provider: {provider_id}")))?;

        registration.auth_provider.start_login().await
    }

    /// Complete a login flow for the given provider.
    ///
    /// On success the resulting [`AuthSession`] is persisted to the credential
    /// store and an [`AccountRecord`] is created in the account store.
    pub async fn complete_login(
        &self,
        provider_id: &ProviderId,
        params: &Metadata,
    ) -> Result<AuthSession> {
        let registration = self
            .provider_registry
            .get(provider_id)
            .ok_or_else(|| FrameworkError::auth(format!("unknown provider: {provider_id}")))?;

        let AuthCompletion { session } =
            registration.auth_provider.complete_login(params).await?;

        // Persist the auth session.
        self.credential_store
            .set_auth_session(provider_id, &session)
            .await?;

        // Persist an account record.
        let account = AccountRecord {
            provider_id: provider_id.clone(),
            display_name: registration.descriptor.display_name.clone(),
            created_at: Utc::now(),
            metadata: Metadata::new(),
        };
        self.account_store.save_account(&account).await?;

        Ok(session)
    }

    /// Log out of the given provider, clearing stored credentials and the
    /// account record.
    pub async fn logout(&self, provider_id: &ProviderId) -> Result<()> {
        let registration = self
            .provider_registry
            .get(provider_id)
            .ok_or_else(|| FrameworkError::auth(format!("unknown provider: {provider_id}")))?;

        // If we have a persisted session, tell the auth provider about it so it
        // can revoke tokens, etc.
        if let Some(session) = self.credential_store.get_auth_session(provider_id).await? {
            registration.auth_provider.logout(&session).await?;
        }

        // Clear stored credentials.
        self.credential_store
            .clear_auth_session(provider_id)
            .await?;
        self.credential_store.clear_api_key(provider_id).await?;

        // Remove the account record.
        self.account_store.delete_account(provider_id).await?;

        Ok(())
    }

    /// Attempt to discover an existing auth session for the given provider
    /// from the credential store.
    ///
    /// Returns `None` if no session is stored or if the stored session fails
    /// validation.
    pub async fn discover_session(
        &self,
        provider_id: &ProviderId,
    ) -> Result<Option<AuthSession>> {
        let registration = self
            .provider_registry
            .get(provider_id)
            .ok_or_else(|| FrameworkError::auth(format!("unknown provider: {provider_id}")))?;

        let session = match self.credential_store.get_auth_session(provider_id).await? {
            Some(s) => s,
            None => return Ok(None),
        };

        // Validate the stored session.
        let valid = registration.auth_provider.validate(&session).await?;
        if valid {
            Ok(Some(session))
        } else {
            Ok(None)
        }
    }

    /// List all stored account records.
    pub async fn list_accounts(&self) -> Result<Vec<AccountRecord>> {
        self.account_store.list_accounts().await
    }

    /// Return a summary of what credentials are available for a provider.
    pub async fn credential_status(
        &self,
        provider_id: &ProviderId,
    ) -> Result<CredentialStatus> {
        self.credential_store.credential_status(provider_id).await
    }
}
