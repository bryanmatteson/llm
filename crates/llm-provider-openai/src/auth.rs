use base64::Engine;
use llm_auth::{
    AuthCompletion, AuthMethod, AuthProvider, AuthSession, AuthStart, OAuthEndpoints,
    OAuthTokenResponse, PkceChallenge, RedirectStrategy, TokenPair, build_auth_url, generate_state,
    redirect_uri_for,
};
use llm_core::{FrameworkError, Metadata, ProviderId};

use crate::descriptor::{OPENAI_CLIENT_ID, PROVIDER_ID};

/// Fixed local port for the OAuth redirect listener.
const REDIRECT_PORT: u16 = 1455;

/// Path portion of the redirect URI.
const REDIRECT_PATH: &str = "/auth/callback";

/// OAuth endpoint configuration for OpenAI.
fn openai_endpoints() -> OAuthEndpoints {
    OAuthEndpoints {
        auth_url: "https://auth.openai.com/oauth/authorize",
        token_url: "https://auth.openai.com/oauth/token",
        scopes: "openid profile email offline_access",
        default_client_id: Some(OPENAI_CLIENT_ID),
        default_client_secret: None,
        redirect: RedirectStrategy::Localhost {
            path: REDIRECT_PATH,
            fixed_port: Some(REDIRECT_PORT),
        },
        extra_auth_params: &[],
        state_is_verifier: false,
    }
}

/// Extract the `chatgpt_account_id` from an OpenAI OAuth access token (JWT).
///
/// The access token contains a custom claim namespace
/// `https://api.openai.com/auth` with a `chatgpt_account_id` field.
/// Returns `None` if the token is not a valid JWT or lacks the claim.
fn extract_chatgpt_account_id(access_token: &str) -> Option<String> {
    let payload_b64 = access_token.split('.').nth(1)?;
    let payload_bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload_b64)
        .ok()?;
    let claims: serde_json::Value = serde_json::from_slice(&payload_bytes).ok()?;
    claims
        .get("https://api.openai.com/auth")?
        .get("chatgpt_account_id")?
        .as_str()
        .map(String::from)
}

/// Build an [`AuthSession`] metadata map, optionally including the
/// ChatGPT account ID extracted from the access token JWT.
fn build_oauth_metadata(access_token: &str) -> Metadata {
    let mut metadata = Metadata::new();
    if let Some(account_id) = extract_chatgpt_account_id(access_token) {
        metadata.insert("chatgpt_account_id".to_string(), account_id);
    }
    metadata
}

/// Authentication provider for OpenAI.
///
/// Supports two discovery paths:
///
/// 1. If the `OPENAI_API_KEY` environment variable is set, an API-key auth
///    method is returned.
/// 2. Otherwise, an OAuth browser flow is offered.
pub struct OpenAiAuthProvider {
    provider_id: ProviderId,
    http: reqwest::Client,
    /// Held between `start_login` and `complete_login` so the verifier is
    /// available for the token exchange.
    pkce: std::sync::Mutex<Option<PkceChallenge>>,
}

impl std::fmt::Debug for OpenAiAuthProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiAuthProvider")
            .field("provider_id", &self.provider_id)
            .finish_non_exhaustive()
    }
}

impl OpenAiAuthProvider {
    pub fn new() -> Self {
        Self {
            provider_id: PROVIDER_ID.clone(),
            http: reqwest::Client::new(),
            pkce: std::sync::Mutex::new(None),
        }
    }
}

impl Default for OpenAiAuthProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl AuthProvider for OpenAiAuthProvider {
    fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    async fn discover(&self) -> Result<Vec<AuthMethod>, FrameworkError> {
        let mut methods = Vec::new();

        // Check for an existing API key in the environment.
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
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
        let endpoints = openai_endpoints();
        let pkce = PkceChallenge::generate();
        let state = generate_state();
        let redirect_uri = redirect_uri_for(&endpoints.redirect, REDIRECT_PORT);

        let url = build_auth_url(&endpoints, OPENAI_CLIENT_ID, &redirect_uri, &pkce, &state)?;

        // Stash the PKCE verifier so `complete_login` can use it.
        *self.pkce.lock().unwrap() = Some(pkce);

        Ok(AuthStart::OAuthBrowser {
            url,
            redirect_uri,
            state,
        })
    }

    async fn complete_login(&self, params: &Metadata) -> Result<AuthCompletion, FrameworkError> {
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
        let _state = params
            .get("state")
            .ok_or_else(|| FrameworkError::auth("missing \"state\" parameter"))?;

        let pkce = self.pkce.lock().unwrap().take().ok_or_else(|| {
            FrameworkError::auth("no PKCE challenge found; was start_login called?")
        })?;

        let endpoints = openai_endpoints();
        let redirect_uri = redirect_uri_for(&endpoints.redirect, REDIRECT_PORT);

        let token_resp: OAuthTokenResponse = self
            .http
            .post(endpoints.token_url)
            .form(&[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", &redirect_uri),
                ("client_id", OPENAI_CLIENT_ID),
                ("code_verifier", &pkce.verifier),
            ])
            .send()
            .await
            .map_err(|e| FrameworkError::auth(format!("token exchange request failed: {e}")))?
            .error_for_status()
            .map_err(|e| FrameworkError::auth(format!("token exchange returned error: {e}")))?
            .json()
            .await
            .map_err(|e| FrameworkError::auth(format!("failed to parse token response: {e}")))?;

        let expires_in = token_resp.expires_in.unwrap_or(3600) as i64;

        let metadata = build_oauth_metadata(&token_resp.access_token);

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
            metadata,
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

        let endpoints = openai_endpoints();

        let token_resp: OAuthTokenResponse = self
            .http
            .post(endpoints.token_url)
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("client_id", OPENAI_CLIENT_ID),
            ])
            .send()
            .await
            .map_err(|e| FrameworkError::auth(format!("refresh request failed: {e}")))?
            .error_for_status()
            .map_err(|e| FrameworkError::auth(format!("refresh returned error: {e}")))?
            .json()
            .await
            .map_err(|e| FrameworkError::auth(format!("failed to parse refresh response: {e}")))?;

        let expires_in = token_resp.expires_in.unwrap_or(3600) as i64;

        // Re-extract account ID from the refreshed token; fall back to the
        // previous session's metadata if the new token lacks the claim.
        let mut metadata = build_oauth_metadata(&token_resp.access_token);
        if !metadata.contains_key("chatgpt_account_id") {
            if let Some(prev) = session.metadata.get("chatgpt_account_id") {
                metadata.insert("chatgpt_account_id".to_string(), prev.clone());
            }
        }

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
            metadata,
        })
    }

    async fn validate(&self, session: &AuthSession) -> Result<bool, FrameworkError> {
        Ok(!session.tokens.is_expired())
    }
}
