use std::sync::Arc;

use llm_auth::AuthSession;
use llm_config::SessionDefaults;
use llm_core::{FrameworkError, Message, ProviderId, Result, SessionId};
use llm_session::{
    AutoApproveHandler, EventReceiver, EventSender, SessionConfig, SessionHandle, SessionManager,
    TurnLoopContext, TurnOutcome, event_channel, run_turn_loop,
};
use llm_tools::{ToolPolicy, ToolRegistry};

use crate::registry::ProviderRegistry;

/// High-level session service that creates sessions and drives the turn loop.
pub struct SessionService {
    provider_registry: Arc<ProviderRegistry>,
    session_manager: Arc<dyn SessionManager>,
    tool_registry: Arc<ToolRegistry>,
    /// Session defaults from the application config (if loaded).
    session_defaults: Option<SessionDefaults>,
    /// Default tool policy derived from the application config (if loaded).
    default_tool_policy: Option<ToolPolicy>,
    /// Default provider from the application config (if set).
    default_provider: Option<ProviderId>,
    /// Level 1 skill metadata prompt injected into new sessions.
    skill_metadata_prompt: Option<String>,
}

impl SessionService {
    /// Create a new `SessionService`.
    pub fn new(
        provider_registry: Arc<ProviderRegistry>,
        session_manager: Arc<dyn SessionManager>,
        tool_registry: Arc<ToolRegistry>,
        session_defaults: Option<SessionDefaults>,
        default_tool_policy: Option<ToolPolicy>,
        default_provider: Option<ProviderId>,
        skill_metadata_prompt: Option<String>,
    ) -> Self {
        Self {
            provider_registry,
            session_manager,
            tool_registry,
            session_defaults,
            default_tool_policy,
            default_provider,
            skill_metadata_prompt,
        }
    }

    /// Returns the default provider ID from the application config, if set.
    pub fn default_provider(&self) -> Option<&ProviderId> {
        self.default_provider.as_ref()
    }

    /// Returns the session defaults from the application config, if loaded.
    pub fn session_defaults(&self) -> Option<&SessionDefaults> {
        self.session_defaults.as_ref()
    }

    /// Returns the default tool policy derived from config, if loaded.
    pub fn default_tool_policy(&self) -> Option<&ToolPolicy> {
        self.default_tool_policy.as_ref()
    }

    /// Create a new conversation session.
    ///
    /// Returns a [`SessionHandle`], an [`EventSender`] that the caller must
    /// keep alive for the duration of the session, and an [`EventReceiver`]
    /// for observing session progress.
    pub async fn create_session(
        &self,
        provider_id: &ProviderId,
        auth: &AuthSession,
        config: SessionConfig,
    ) -> Result<(SessionHandle, EventSender, EventReceiver)> {
        let config = self.inject_skill_metadata(config);

        // Verify the provider exists and create a client to prove the auth is
        // valid. We hold the client in the handle conceptually, but since
        // `SessionHandle` does not store a client we just validate here.
        let registration = self
            .provider_registry
            .get(provider_id)
            .ok_or_else(|| FrameworkError::session(format!("unknown provider: {provider_id}")))?;

        let model = config
            .model
            .clone()
            .unwrap_or_else(|| registration.descriptor.default_model.clone());

        // Validate that a client can be created (ensures auth is valid).
        let _client = registration
            .client_factory
            .create_client(auth, &model)
            .await?;

        let handle = self.session_manager.create_session(config).await?;
        let (tx, rx) = event_channel();

        Ok((handle, tx, rx))
    }

    fn inject_skill_metadata(&self, mut config: SessionConfig) -> SessionConfig {
        let Some(skill_metadata_prompt) = self.skill_metadata_prompt.as_deref() else {
            return config;
        };

        config.system_prompt = Some(match config.system_prompt.take() {
            Some(system_prompt) if !system_prompt.trim().is_empty() => {
                format!("{system_prompt}\n\n{skill_metadata_prompt}")
            }
            _ => skill_metadata_prompt.to_string(),
        });

        config
    }

    /// Send a user message to an existing session and run the turn loop to
    /// completion.
    ///
    /// The message is appended to the session's conversation, then the turn
    /// loop drives the provider interaction (including tool calls) until the
    /// model produces a final text response.
    pub async fn send_message(
        &self,
        session_id: &SessionId,
        auth: &AuthSession,
        text: &str,
    ) -> Result<TurnOutcome> {
        let mut handle = self
            .session_manager
            .get_session(session_id)
            .await?
            .ok_or_else(|| FrameworkError::session(format!("session not found: {session_id}")))?;

        let provider_id = &handle.config.provider_id;
        let registration = self
            .provider_registry
            .get(provider_id)
            .ok_or_else(|| FrameworkError::session(format!("unknown provider: {provider_id}")))?;

        let model = handle
            .config
            .model
            .clone()
            .unwrap_or_else(|| registration.descriptor.default_model.clone());

        let client = registration
            .client_factory
            .create_client(auth, &model)
            .await?;

        // Append the user message to the conversation.
        handle.conversation.append_message(Message::user(text));

        let (tx, _rx) = event_channel();
        let approval_handler = AutoApproveHandler;

        let outcome = run_turn_loop(TurnLoopContext {
            session_id,
            client: client.as_ref(),
            conversation: &mut handle.conversation,
            tool_registry: &self.tool_registry,
            tool_adapter: registration.tool_adapter.as_ref(),
            config: &handle.config,
            approval_handler: &approval_handler,
            event_tx: Some(&tx),
        })
        .await?;

        // Accumulate this turn's usage into the session's lifetime total.
        handle.total_usage.accumulate(&outcome.usage);

        Ok(outcome)
    }

    /// List the ids of all known sessions.
    pub async fn list_sessions(&self) -> Result<Vec<SessionId>> {
        self.session_manager.list_sessions().await
    }
}
