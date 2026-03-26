use std::sync::Arc;

use llm_core::ProviderDescriptor;
use llm_provider_openai::{OpenAiAuthProvider, provider_descriptor};
use llm_session::{DefaultSessionManager, SessionManager};
use llm_store::{
    AccountStore, CredentialStore, InMemoryAccountStore, InMemoryCredentialStore,
    InMemorySessionStore, SessionStore,
};
use llm_tools::ToolRegistry;

use llm_auth::AuthProvider;

/// Aggregated application context wired up by [`build_default_context`].
///
/// Holds shared references to stores, registries, and provider-specific
/// helpers that the command handlers need.
pub struct AppContext {
    pub credential_store: Arc<dyn CredentialStore>,
    pub account_store: Arc<dyn AccountStore>,
    #[allow(dead_code)]
    pub session_store: Arc<dyn SessionStore>,
    pub session_manager: Arc<dyn SessionManager>,
    pub tool_registry: Arc<ToolRegistry>,
    pub auth_providers: Vec<Arc<dyn AuthProvider>>,
    pub provider_descriptors: Vec<ProviderDescriptor>,
}

impl AppContext {
    /// Look up an auth provider by its string identifier.
    pub fn auth_provider(&self, name: &str) -> Option<Arc<dyn AuthProvider>> {
        self.auth_providers
            .iter()
            .find(|p| p.provider_id().as_str() == name)
            .cloned()
    }

    /// Look up a provider descriptor by its string identifier.
    pub fn provider_descriptor(&self, name: &str) -> Option<&ProviderDescriptor> {
        self.provider_descriptors
            .iter()
            .find(|d| d.id.as_str() == name)
    }
}

/// Build the default [`AppContext`] using in-memory stores and the OpenAI
/// provider.
pub fn build_default_context() -> llm_core::Result<AppContext> {
    // -- Stores --
    let credential_store: Arc<dyn CredentialStore> = Arc::new(InMemoryCredentialStore::new());
    let account_store: Arc<dyn AccountStore> = Arc::new(InMemoryAccountStore::new());
    let session_store: Arc<dyn SessionStore> = Arc::new(InMemorySessionStore::new());

    // -- Session manager --
    let session_manager: Arc<dyn SessionManager> =
        Arc::new(DefaultSessionManager::new(session_store.clone()));

    // -- Tool registry (empty for now; tools are registered by plugins) --
    let tool_registry = Arc::new(ToolRegistry::new());

    // -- Auth providers --
    let openai_auth: Arc<dyn AuthProvider> = Arc::new(OpenAiAuthProvider::new());
    let auth_providers = vec![openai_auth];

    // -- Provider descriptors --
    let provider_descriptors = vec![provider_descriptor()];

    Ok(AppContext {
        credential_store,
        account_store,
        session_store,
        session_manager,
        tool_registry,
        auth_providers,
        provider_descriptors,
    })
}
