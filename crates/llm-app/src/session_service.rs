use std::sync::Arc;

use llm_auth::AuthSession;
use llm_core::{FrameworkError, Message, ProviderId, Result, SessionId};
use llm_session::{
    AutoApproveHandler, EventReceiver, EventSender, SessionConfig, SessionHandle, SessionManager,
    TurnOutcome, event_channel, run_turn_loop,
};
use llm_tools::ToolRegistry;

use crate::registry::ProviderRegistry;

/// High-level session service that creates sessions and drives the turn loop.
pub struct SessionService {
    provider_registry: Arc<ProviderRegistry>,
    session_manager: Arc<dyn SessionManager>,
    tool_registry: Arc<ToolRegistry>,
}

impl SessionService {
    /// Create a new `SessionService`.
    pub fn new(
        provider_registry: Arc<ProviderRegistry>,
        session_manager: Arc<dyn SessionManager>,
        tool_registry: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            provider_registry,
            session_manager,
            tool_registry,
        }
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
        // Verify the provider exists and create a client to prove the auth is
        // valid. We hold the client in the handle conceptually, but since
        // `SessionHandle` does not store a client we just validate here.
        let registration = self
            .provider_registry
            .get(provider_id)
            .ok_or_else(|| {
                FrameworkError::session(format!("unknown provider: {provider_id}"))
            })?;

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
            .ok_or_else(|| {
                FrameworkError::session(format!("session not found: {session_id}"))
            })?;

        let provider_id = &handle.config.provider_id;
        let registration = self
            .provider_registry
            .get(provider_id)
            .ok_or_else(|| {
                FrameworkError::session(format!("unknown provider: {provider_id}"))
            })?;

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
        handle
            .conversation
            .append_message(Message::user(text));

        let (tx, _rx) = event_channel();
        let approval_handler = AutoApproveHandler;

        let outcome = run_turn_loop(
            session_id,
            client.as_ref(),
            &mut handle.conversation,
            &self.tool_registry,
            registration.tool_adapter.as_ref(),
            &handle.config,
            &approval_handler,
            Some(&tx),
        )
        .await?;

        Ok(outcome)
    }

    /// List the ids of all known sessions.
    pub async fn list_sessions(&self) -> Result<Vec<SessionId>> {
        self.session_manager.list_sessions().await
    }
}
