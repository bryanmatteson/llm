use std::sync::Arc;

use llm_config::LlmConfig;

use crate::auth_service::AuthService;
use crate::questionnaire_service::QuestionnaireService;
use crate::registry::ProviderRegistry;
use crate::session_service::SessionService;
use crate::tool_service::ToolService;

/// The top-level application context grouping every service.
///
/// An `AppContext` is built once via [`AppBuilder`](crate::builder::AppBuilder)
/// and then shared (typically behind an `Arc`) with every component of the
/// application.
pub struct LlmContext {
    /// Authentication service for login/logout/discovery.
    pub auth: AuthService,
    /// Session service for creating sessions and driving the turn loop.
    pub sessions: SessionService,
    /// Questionnaire service for setup flows.
    pub questionnaires: QuestionnaireService,
    /// Tool service for inspecting and registering tools.
    pub tools: ToolService,
    /// The shared provider registry.
    pub providers: Arc<ProviderRegistry>,
    /// Optional application config loaded from file or supplied programmatically.
    pub config: Option<LlmConfig>,
}

impl std::fmt::Debug for LlmContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppContext")
            .field("providers", &self.providers.list_provider_ids().len())
            .finish_non_exhaustive()
    }
}
