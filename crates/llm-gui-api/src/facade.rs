//! The [`GuiFacade`] orchestrates application-layer services behind a single,
//! DTO-oriented API designed for GUI consumption.
//!
//! Every method returns lightweight DTOs from [`crate::dto`] so that the GUI
//! layer never needs to depend on internal framework types.

use std::sync::Arc;

use serde_json::json;

use llm_app::AppContext;
use llm_core::{FrameworkError, ModelId, ProviderId, ProviderDescriptor, Result};

use crate::dto::{AuthStatusDto, EventDto, ProviderDto, SessionDto, ToolDto};
use crate::events::SessionEventAdapter;

// ---------------------------------------------------------------------------
// GuiFacade
// ---------------------------------------------------------------------------

/// A high-level facade that GUI controllers call into.
///
/// Every method maps cleanly onto a single GUI action (button press, screen
/// load, etc.) and returns DTOs that are ready to serialise over the wire.
#[derive(Clone)]
pub struct GuiFacade {
    ctx: Arc<AppContext>,
}

impl GuiFacade {
    /// Create a new facade backed by the given application context.
    pub fn new(ctx: Arc<AppContext>) -> Self {
        Self { ctx }
    }

    // -- Providers ---------------------------------------------------------

    /// Return every registered provider as a [`ProviderDto`].
    pub async fn list_providers(&self) -> Result<Vec<ProviderDto>> {
        let descriptors = self.ctx.providers.list_providers();
        Ok(descriptors.into_iter().map(provider_to_dto).collect())
    }

    // -- Auth -------------------------------------------------------------

    /// Return the authentication status for a single provider.
    pub async fn auth_status(&self, provider_id: &str) -> Result<AuthStatusDto> {
        let pid = ProviderId::new(provider_id);
        let session = self.ctx.auth.discover_session(&pid).await?;
        let (authenticated, method) = match &session {
            Some(s) => {
                let method_label = format!("{:?}", s.method);
                (true, Some(method_label))
            }
            None => (false, None),
        };
        Ok(AuthStatusDto {
            provider_id: provider_id.to_owned(),
            authenticated,
            method,
        })
    }

    /// Begin a login flow for the given provider.
    ///
    /// Returns the `AuthStart` payload serialised as JSON so the GUI can
    /// decide how to present the flow (open browser, show device code, etc.).
    pub async fn start_login(&self, provider_id: &str) -> Result<serde_json::Value> {
        let pid = ProviderId::new(provider_id);
        let auth_start = self.ctx.auth.start_login(&pid).await?;

        // `AuthStart` is not `Serialize`, so we manually build a JSON value
        // from the variant.
        let value = auth_start_to_json(&auth_start);
        Ok(value)
    }

    /// Complete a login flow with parameters supplied by the GUI.
    pub async fn complete_login(
        &self,
        provider_id: &str,
        params: serde_json::Value,
    ) -> Result<AuthStatusDto> {
        let pid = ProviderId::new(provider_id);

        // Convert the JSON params into a Metadata map for the auth layer.
        let metadata = json_to_metadata(&params)?;
        let _session = self.ctx.auth.complete_login(&pid, &metadata).await?;

        // Return the refreshed auth status.
        self.auth_status(provider_id).await
    }

    // -- Sessions ---------------------------------------------------------

    /// List all sessions as lightweight DTOs.
    pub async fn list_sessions(&self) -> Result<Vec<SessionDto>> {
        let session_ids = self.ctx.sessions.list_sessions().await?;
        Ok(session_ids
            .into_iter()
            .map(|id| SessionDto {
                id: id.to_string(),
                provider_id: String::new(),
                model: String::new(),
                message_count: 0,
            })
            .collect())
    }

    /// Create a new session for the given provider (and optional model).
    ///
    /// This requires that the provider has a valid authenticated session.
    pub async fn create_session(
        &self,
        provider_id: &str,
        model: Option<&str>,
    ) -> Result<SessionDto> {
        let pid = ProviderId::new(provider_id);

        let auth_session = self
            .ctx
            .auth
            .discover_session(&pid)
            .await?
            .ok_or_else(|| FrameworkError::auth("not authenticated"))?;

        let config = llm_session::SessionConfig {
            provider_id: pid.clone(),
            model: model.map(ModelId::new),
            system_prompt: None,
            tool_policy: Default::default(),
            limits: Default::default(),
            metadata: Default::default(),
        };

        let (handle, _rx) = self
            .ctx
            .sessions
            .create_session(&pid, &auth_session, config)
            .await?;

        let model_name = handle
            .config
            .model
            .as_ref()
            .map_or_else(String::new, |m| m.to_string());

        Ok(SessionDto {
            id: handle.id.to_string(),
            provider_id: provider_id.to_owned(),
            model: model_name,
            message_count: handle.conversation.len(),
        })
    }

    /// Send a user message to an existing session and return the resulting
    /// event as an [`EventDto`].
    ///
    /// The caller must ensure the user is authenticated with the provider
    /// that owns the session.
    pub async fn send_message(&self, session_id: &str, text: &str) -> Result<EventDto> {
        let sid = llm_core::SessionId::new(session_id);

        // We need an auth session for the provider that owns this session.
        // Since we only have the session id, we look up the session to
        // discover its provider, then discover the auth session.
        //
        // For the MVP, we pass a placeholder approach: the caller should
        // have already authenticated. We look up all accounts and find the
        // matching provider from the session config.
        //
        // In a more complete implementation, the session would be looked up
        // first to get the provider id, then the auth session would be
        // discovered. For now we need a two-step approach.
        //
        // We ask the session_manager (via sessions service) for the handle,
        // but the SessionService::send_message already does internal lookup,
        // so we need to provide the AuthSession externally.

        // Discover all accounts to find a valid auth session.
        let accounts = self.ctx.auth.list_accounts().await?;
        if accounts.is_empty() {
            return Err(FrameworkError::auth("no authenticated accounts"));
        }

        // Try each account until we find one that works.
        let mut last_error = None;
        for account in &accounts {
            if let Ok(Some(auth_session)) =
                self.ctx.auth.discover_session(&account.provider_id).await
            {
                match self
                    .ctx
                    .sessions
                    .send_message(&sid, &auth_session, text)
                    .await
                {
                    Ok(outcome) => {
                        let event_json = turn_outcome_to_json(&outcome);
                        return Ok(SessionEventAdapter::adapt_event(&event_json));
                    }
                    Err(e) => {
                        last_error = Some(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| FrameworkError::auth("no valid auth session found")))
    }

    // -- Tools ------------------------------------------------------------

    /// List every registered tool as a [`ToolDto`].
    pub async fn list_tools(&self) -> Result<Vec<ToolDto>> {
        let descriptors = self.ctx.tools.list_tools();
        Ok(descriptors
            .into_iter()
            .map(|d| ToolDto {
                id: d.id.to_string(),
                display_name: d.display_name,
                description: d.description,
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Internal mapping helpers
// ---------------------------------------------------------------------------

fn provider_to_dto(desc: &ProviderDescriptor) -> ProviderDto {
    ProviderDto {
        id: desc.id.to_string(),
        display_name: desc.display_name.clone(),
        capabilities: desc
            .capabilities
            .iter()
            .map(|c| format!("{c:?}"))
            .collect(),
    }
}

/// Convert a `TurnOutcome` into the same JSON shape as a `SessionEvent::TurnCompleted`.
///
/// `TurnOutcome` does not implement `Serialize`, so we build the JSON by hand
/// to match the externally-tagged format expected by [`SessionEventAdapter`].
fn turn_outcome_to_json(outcome: &llm_session::TurnOutcome) -> serde_json::Value {
    json!({
        "TurnCompleted": {
            "text": outcome.final_text,
            "model": outcome.model.as_str(),
            "usage": {
                "input_tokens": outcome.usage.input_tokens,
                "output_tokens": outcome.usage.output_tokens,
            }
        }
    })
}

/// Convert an `AuthStart` enum into a JSON value.
///
/// `AuthStart` is not `Serialize` so we build the JSON manually, mirroring
/// the variant names used by `llm-auth`.
fn auth_start_to_json(start: &llm_auth::AuthStart) -> serde_json::Value {
    match start {
        llm_auth::AuthStart::OAuthBrowser {
            url,
            redirect_uri,
            state,
        } => json!({
            "OAuthBrowser": {
                "url": url,
                "redirect_uri": redirect_uri,
                "state": state,
            }
        }),
        llm_auth::AuthStart::DeviceCode {
            verification_uri,
            user_code,
            interval,
        } => json!({
            "DeviceCode": {
                "verification_uri": verification_uri,
                "user_code": user_code,
                "interval": interval,
            }
        }),
        llm_auth::AuthStart::ApiKeyPrompt { env_var_hint } => json!({
            "ApiKeyPrompt": {
                "env_var_hint": env_var_hint,
            }
        }),
    }
}

/// Convert a JSON value (typically an object) into a `Metadata` (BTreeMap).
///
/// String values are kept as-is; other JSON types are serialised to their
/// compact JSON string form.
fn json_to_metadata(
    value: &serde_json::Value,
) -> Result<llm_core::Metadata> {
    let mut map = llm_core::Metadata::new();
    if let Some(obj) = value.as_object() {
        for (k, v) in obj {
            let s = match v.as_str() {
                Some(s) => s.to_owned(),
                None => v.to_string(),
            };
            map.insert(k.clone(), s);
        }
    }
    Ok(map)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn auth_start_to_json_oauth() {
        let start = llm_auth::AuthStart::OAuthBrowser {
            url: "https://example.com/auth".into(),
            redirect_uri: "http://localhost:8080/callback".into(),
            state: "abc123".into(),
        };
        let value = auth_start_to_json(&start);
        assert_eq!(value["OAuthBrowser"]["url"], "https://example.com/auth");
        assert_eq!(value["OAuthBrowser"]["state"], "abc123");
    }

    #[test]
    fn auth_start_to_json_device_code() {
        let start = llm_auth::AuthStart::DeviceCode {
            verification_uri: "https://example.com/device".into(),
            user_code: "ABCD-1234".into(),
            interval: 5,
        };
        let value = auth_start_to_json(&start);
        assert_eq!(value["DeviceCode"]["user_code"], "ABCD-1234");
        assert_eq!(value["DeviceCode"]["interval"], 5);
    }

    #[test]
    fn auth_start_to_json_api_key() {
        let start = llm_auth::AuthStart::ApiKeyPrompt {
            env_var_hint: "OPENAI_API_KEY".into(),
        };
        let value = auth_start_to_json(&start);
        assert_eq!(value["ApiKeyPrompt"]["env_var_hint"], "OPENAI_API_KEY");
    }

    #[test]
    fn json_to_metadata_string_values() {
        let input = json!({
            "api_key": "sk-test",
            "code": "auth-code-123"
        });
        let meta = json_to_metadata(&input).unwrap();
        assert_eq!(meta.get("api_key").unwrap(), "sk-test");
        assert_eq!(meta.get("code").unwrap(), "auth-code-123");
    }

    #[test]
    fn json_to_metadata_non_string_values() {
        let input = json!({
            "count": 42,
            "enabled": true
        });
        let meta = json_to_metadata(&input).unwrap();
        assert_eq!(meta.get("count").unwrap(), "42");
        assert_eq!(meta.get("enabled").unwrap(), "true");
    }

    #[test]
    fn json_to_metadata_non_object() {
        let input = json!("just a string");
        let meta = json_to_metadata(&input).unwrap();
        assert!(meta.is_empty());
    }

    #[test]
    fn provider_to_dto_mapping() {
        let desc = ProviderDescriptor {
            id: ProviderId::new("openai"),
            display_name: "OpenAI".into(),
            default_model: ModelId::new("gpt-4o"),
            capabilities: vec![
                llm_core::ProviderCapability::OAuth,
                llm_core::ProviderCapability::Streaming,
            ],
            metadata: Default::default(),
        };
        let dto = provider_to_dto(&desc);
        assert_eq!(dto.id, "openai");
        assert_eq!(dto.display_name, "OpenAI");
        assert_eq!(dto.capabilities.len(), 2);
        assert!(dto.capabilities.contains(&"OAuth".to_string()));
        assert!(dto.capabilities.contains(&"Streaming".to_string()));
    }
}
