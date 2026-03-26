use llm_auth::{
    AuthCompletion, AuthMethod, AuthProvider, AuthSession, AuthStart, OAuthEndpoints,
    OAuthTokenResponse, PkceChallenge, RedirectStrategy, TokenPair, build_auth_url,
    redirect_uri_for,
};
use llm_core::{FrameworkError, Metadata, ProviderId};

use crate::descriptor::{ANTHROPIC_CLIENT_ID, PROVIDER_ID};

/// OAuth endpoint configuration for Anthropic.
///
/// Notable differences from OpenAI:
/// - Uses a remote callback redirect (`https://console.anthropic.com/oauth/code/callback`)
/// - The `state` parameter doubles as the PKCE verifier (`state_is_verifier = true`)
/// - Includes extra auth param `("code", "true")`
fn anthropic_endpoints() -> OAuthEndpoints {
    OAuthEndpoints {
        auth_url: "https://claude.ai/oauth/authorize",
        token_url: "https://console.anthropic.com/v1/oauth/token",
        scopes: "org:create_api_key user:profile user:inference",
        default_client_id: Some(ANTHROPIC_CLIENT_ID),
        default_client_secret: None,
        redirect: RedirectStrategy::RemoteCallback {
            redirect_uri: "https://console.anthropic.com/oauth/code/callback",
        },
        extra_auth_params: &[("code", "true")],
        state_is_verifier: true,
    }
}

/// Authentication provider for Anthropic.
///
/// Supports two discovery paths:
///
/// 1. If the `ANTHROPIC_API_KEY` environment variable is set, an API-key auth
///    method is returned.
/// 2. Otherwise, an OAuth browser flow is offered.
pub struct AnthropicAuthProvider {
    provider_id: ProviderId,
    http: reqwest::Client,
    /// Held between `start_login` and `complete_login` so the verifier is
    /// available for the token exchange.
    pkce: std::sync::Mutex<Option<PkceChallenge>>,
}

impl std::fmt::Debug for AnthropicAuthProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicAuthProvider")
            .field("provider_id", &self.provider_id)
            .finish_non_exhaustive()
    }
}

impl AnthropicAuthProvider {
    pub fn new() -> Self {
        Self {
            provider_id: PROVIDER_ID.clone(),
            http: reqwest::Client::new(),
            pkce: std::sync::Mutex::new(None),
        }
    }
}

impl Default for AnthropicAuthProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl AuthProvider for AnthropicAuthProvider {
    fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    async fn discover(&self) -> Result<Vec<AuthMethod>, FrameworkError> {
        let mut methods = Vec::new();

        // Check for an existing API key in the environment.
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            if !key.is_empty() {
                let masked = if key.len() > 8 {
                    format!("{}...{}", &key[..4], &key[key.len() - 4..])
                } else {
                    "****".to_owned()
                };
                methods.push(AuthMethod::ApiKey { masked });
            }
        }

        // OAuth is always available as a fallback.
        methods.push(AuthMethod::OAuth {
            expires_at: chrono::Utc::now(),
        });

        Ok(methods)
    }

    async fn start_login(&self) -> Result<AuthStart, FrameworkError> {
        let endpoints = anthropic_endpoints();
        let pkce = PkceChallenge::generate();

        // Anthropic convention: state parameter = PKCE verifier
        let state = pkce.verifier.clone();

        let redirect_uri = redirect_uri_for(&endpoints.redirect, 0);

        let url = build_auth_url(
            &endpoints,
            ANTHROPIC_CLIENT_ID,
            &redirect_uri,
            &pkce,
            &state,
        )?;

        // Stash the PKCE challenge so `complete_login` can use it.
        *self.pkce.lock().unwrap() = Some(pkce);

        Ok(AuthStart::OAuthBrowser {
            url,
            redirect_uri,
            state,
        })
    }

    async fn complete_login(
        &self,
        params: &Metadata,
    ) -> Result<AuthCompletion, FrameworkError> {
        // ── API-key path ────────────────────────────────────────────
        if let Some(api_key) = params.get("api_key") {
            let masked = if api_key.len() > 8 {
                format!("{}...{}", &api_key[..4], &api_key[api_key.len() - 4..])
            } else {
                "****".to_owned()
            };

            let session = AuthSession {
                provider_id: self.provider_id.clone(),
                method: AuthMethod::ApiKey {
                    masked: masked.clone(),
                },
                tokens: TokenPair::new(api_key.clone(), None, 365 * 24 * 3600),
                metadata: Metadata::new(),
            };

            return Ok(AuthCompletion { session });
        }

        // ── OAuth path ──────────────────────────────────────────────
        let code = params
            .get("code")
            .ok_or_else(|| FrameworkError::auth("missing \"code\" parameter"))?;
        let state = params
            .get("state")
            .ok_or_else(|| FrameworkError::auth("missing \"state\" parameter"))?;

        let pkce = self
            .pkce
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| {
                FrameworkError::auth("no PKCE challenge found; was start_login called?")
            })?;

        let endpoints = anthropic_endpoints();
        let redirect_uri = redirect_uri_for(&endpoints.redirect, 0);

        // Anthropic requires the `state` parameter in the token exchange body.
        let token_resp: OAuthTokenResponse = self
            .http
            .post(endpoints.token_url)
            .form(&[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", &redirect_uri),
                ("client_id", ANTHROPIC_CLIENT_ID),
                ("code_verifier", &pkce.verifier),
                ("state", state),
            ])
            .send()
            .await
            .map_err(|e| FrameworkError::auth(format!("token exchange request failed: {e}")))?
            .error_for_status()
            .map_err(|e| FrameworkError::auth(format!("token exchange returned error: {e}")))?
            .json()
            .await
            .map_err(|e| {
                FrameworkError::auth(format!("failed to parse token response: {e}"))
            })?;

        let expires_in = token_resp.expires_in.unwrap_or(3600) as i64;

        let tokens = TokenPair::new(
            token_resp.access_token,
            token_resp.refresh_token,
            expires_in,
        );

        let session = AuthSession {
            provider_id: self.provider_id.clone(),
            method: AuthMethod::OAuth {
                expires_at: tokens.expires_at,
            },
            tokens,
            metadata: Metadata::new(),
        };

        Ok(AuthCompletion { session })
    }

    async fn logout(&self, _session: &AuthSession) -> Result<(), FrameworkError> {
        // Token clearing is handled by the store layer.
        Ok(())
    }

    async fn refresh(&self, session: &AuthSession) -> Result<AuthSession, FrameworkError> {
        let refresh_token = session
            .tokens
            .refresh_token
            .as_deref()
            .ok_or_else(|| FrameworkError::auth("no refresh token available"))?;

        let endpoints = anthropic_endpoints();

        let token_resp: OAuthTokenResponse = self
            .http
            .post(endpoints.token_url)
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("client_id", ANTHROPIC_CLIENT_ID),
            ])
            .send()
            .await
            .map_err(|e| FrameworkError::auth(format!("refresh request failed: {e}")))?
            .error_for_status()
            .map_err(|e| FrameworkError::auth(format!("refresh returned error: {e}")))?
            .json()
            .await
            .map_err(|e| {
                FrameworkError::auth(format!("failed to parse refresh response: {e}"))
            })?;

        let expires_in = token_resp.expires_in.unwrap_or(3600) as i64;

        let tokens = TokenPair::new(
            token_resp.access_token,
            token_resp
                .refresh_token
                .or_else(|| session.tokens.refresh_token.clone()),
            expires_in,
        );

        Ok(AuthSession {
            provider_id: self.provider_id.clone(),
            method: AuthMethod::OAuth {
                expires_at: tokens.expires_at,
            },
            tokens,
            metadata: session.metadata.clone(),
        })
    }

    async fn validate(&self, session: &AuthSession) -> Result<bool, FrameworkError> {
        Ok(!session.tokens.is_expired())
    }
}
