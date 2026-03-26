use std::path::Path;
use std::sync::Arc;

use llm_config::AppConfig;
use llm_core::Result;
use llm_session::{DefaultSessionManager, SessionManager};
use llm_store::{
    AccountStore, CredentialStore, FileAccountStore, FileCredentialStore, FileSessionStore,
    InMemoryAccountStore, InMemoryCredentialStore, InMemorySessionStore, SessionStore,
};
use llm_tools::{DynTool, ToolApproval, ToolPolicyBuilder, ToolRegistry};

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
    tools: Vec<Arc<dyn DynTool>>,
    config: Option<AppConfig>,
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
            config: None,
        }
    }

    /// Use file-backed stores rooted at `dir`.
    ///
    /// Creates subdirectories under `dir` for credentials, accounts, and
    /// sessions.  Directories are created lazily on first write.
    ///
    /// This is a convenience that sets all three stores at once.  You can
    /// still override individual stores afterward with
    /// [`with_credential_store`](Self::with_credential_store), etc.
    pub fn with_data_dir(self, dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        self.with_credential_store(Arc::new(FileCredentialStore::new(dir.join("credentials"))))
            .with_account_store(Arc::new(FileAccountStore::new(dir.join("accounts"))))
            .with_session_store(Arc::new(FileSessionStore::new(dir.join("sessions"))))
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
    pub fn register_tool(mut self, tool: Arc<dyn DynTool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Apply an [`AppConfig`] to this builder.
    ///
    /// This sets session defaults and tool policies from the config file.
    /// Provider registrations and stores must still be set explicitly.
    pub fn with_config(mut self, config: &AppConfig) -> Self {
        self.config = Some(config.clone());
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

        // -- Apply config (if provided) --------------------------------------

        let default_tool_policy = if let Some(ref cfg) = self.config {
            let mut builder = ToolPolicyBuilder::new();
            if cfg.session_defaults.tool_confirmation_required {
                builder = builder.default(ToolApproval::RequireConfirmation);
            }
            for tp in &cfg.tool_policies {
                // The KDL validation layer guarantees `allowed` is always
                // `Some` by the time we reach here.  The parse rejects any
                // tool-policy missing both `allow` and `forbid`.
                let allowed = tp.allowed.unwrap_or(false);
                let approval = if !allowed {
                    ToolApproval::Deny
                } else if tp.require_confirmation {
                    ToolApproval::RequireConfirmation
                } else {
                    ToolApproval::Auto
                };
                builder = builder.rule(
                    tp.tool_id.as_str(),
                    approval,
                    tp.max_calls.map(|n| n as u32),
                );
            }
            Some(builder.build())
        } else {
            None
        };

        // -- Session manager -------------------------------------------------

        let session_manager: Arc<dyn SessionManager> =
            Arc::new(DefaultSessionManager::new(session_store));

        // -- Services --------------------------------------------------------

        let auth = AuthService::new(Arc::clone(&registry), credential_store, account_store);

        let sessions = SessionService::new(
            Arc::clone(&registry),
            session_manager,
            Arc::clone(&tool_registry),
            self.config.as_ref().map(|c| c.session_defaults.clone()),
            default_tool_policy,
            self.config
                .as_ref()
                .and_then(|c| c.default_provider.clone()),
        );

        let questionnaires = QuestionnaireService::new();

        let tools = ToolService::new(tool_registry);

        Ok(AppContext {
            auth,
            sessions,
            questionnaires,
            tools,
            providers: registry,
            config: self.config,
        })
    }
}

impl Default for AppBuilder {
    fn default() -> Self {
        Self::new()
    }
}
