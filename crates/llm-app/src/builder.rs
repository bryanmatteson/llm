use std::sync::Arc;

use llm_core::Result;
use llm_session::{DefaultSessionManager, SessionManager};
use llm_store::{
    AccountStore, CredentialStore, InMemoryAccountStore, InMemoryCredentialStore,
    InMemorySessionStore, SessionStore,
};
use llm_tools::{Tool, ToolRegistry};

use crate::auth_service::AuthService;
use crate::context::AppContext;
use crate::questionnaire_service::QuestionnaireService;
use crate::registry::{ProviderRegistration, ProviderRegistry};
use crate::session_service::SessionService;
use crate::tool_service::ToolService;

/// Builder for constructing an [`AppContext`] with all required services.
///
/// # Example
///
/// ```ignore
/// let ctx = AppBuilder::new()
///     .register_provider(my_registration)
///     .register_tool(my_tool)
///     .build()?;
/// ```
pub struct AppBuilder {
    credential_store: Option<Arc<dyn CredentialStore>>,
    account_store: Option<Arc<dyn AccountStore>>,
    session_store: Option<Arc<dyn SessionStore>>,
    registrations: Vec<ProviderRegistration>,
    tools: Vec<Arc<dyn Tool>>,
}

impl AppBuilder {
    /// Create a new builder with default (in-memory) stores.
    pub fn new() -> Self {
        Self {
            credential_store: None,
            account_store: None,
            session_store: None,
            registrations: Vec::new(),
            tools: Vec::new(),
        }
    }

    /// Supply a custom credential store.
    ///
    /// If not called, an [`InMemoryCredentialStore`] is used.
    pub fn with_credential_store(mut self, store: Arc<dyn CredentialStore>) -> Self {
        self.credential_store = Some(store);
        self
    }

    /// Supply a custom account store.
    ///
    /// If not called, an [`InMemoryAccountStore`] is used.
    pub fn with_account_store(mut self, store: Arc<dyn AccountStore>) -> Self {
        self.account_store = Some(store);
        self
    }

    /// Supply a custom session store.
    ///
    /// If not called, an [`InMemorySessionStore`] is used.
    pub fn with_session_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.session_store = Some(store);
        self
    }

    /// Register a provider with the application.
    pub fn register_provider(mut self, registration: ProviderRegistration) -> Self {
        self.registrations.push(registration);
        self
    }

    /// Register a tool with the application.
    pub fn register_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Consume the builder and produce a fully wired [`AppContext`].
    ///
    /// Returns an error if the resulting configuration is invalid (e.g. no
    /// providers registered).
    pub fn build(self) -> Result<AppContext> {
        // -- Stores ----------------------------------------------------------

        let credential_store: Arc<dyn CredentialStore> = self
            .credential_store
            .unwrap_or_else(|| Arc::new(InMemoryCredentialStore::new()));

        let account_store: Arc<dyn AccountStore> = self
            .account_store
            .unwrap_or_else(|| Arc::new(InMemoryAccountStore::new()));

        let session_store: Arc<dyn SessionStore> = self
            .session_store
            .unwrap_or_else(|| Arc::new(InMemorySessionStore::new()));

        // -- Provider registry -----------------------------------------------

        let mut registry = ProviderRegistry::new();
        for reg in self.registrations {
            registry.register(reg);
        }
        let registry = Arc::new(registry);

        // -- Tool registry ---------------------------------------------------

        let mut tool_registry = ToolRegistry::new();
        for tool in self.tools {
            tool_registry.register(tool);
        }
        let tool_registry = Arc::new(tool_registry);

        // -- Session manager -------------------------------------------------

        let session_manager: Arc<dyn SessionManager> =
            Arc::new(DefaultSessionManager::new(session_store));

        // -- Services --------------------------------------------------------

        let auth = AuthService::new(
            Arc::clone(&registry),
            credential_store,
            account_store,
        );

        let sessions = SessionService::new(
            Arc::clone(&registry),
            session_manager,
            Arc::clone(&tool_registry),
        );

        let questionnaires = QuestionnaireService::new();

        let tools = ToolService::new(tool_registry);

        Ok(AppContext {
            auth,
            sessions,
            questionnaires,
            tools,
            providers: registry,
        })
    }
}

impl Default for AppBuilder {
    fn default() -> Self {
        Self::new()
    }
}
