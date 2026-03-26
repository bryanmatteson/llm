use llm_auth::{
    AuthCompletion, AuthMethod, AuthProvider, AuthSession, AuthStart, OAuthEndpoints,
    OAuthTokenResponse, PkceChallenge, RedirectStrategy, TokenPair, build_auth_url, generate_state,
    redirect_uri_for,
};
use llm_core::{FrameworkError, Metadata, ProviderId};

use crate::descriptor::{GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, PROVIDER_ID};

/// Path portion of the OAuth redirect URI.
const REDIRECT_PATH: &str = "/oauth2callback";

/// OAuth endpoint configuration for Google.
///
/// Notable differences from OpenAI and Anthropic:
/// - Requires a `client_secret` (installed-app secret, safe to embed)
/// - Uses localhost redirect with a dynamic port
/// - Requires extra params `access_type=offline` and `prompt=consent`
///   to obtain a refresh token
/// - Standard CSRF state (not PKCE-as-state like Anthropic)
fn google_endpoints() -> OAuthEndpoints {
    OAuthEndpoints {
        auth_url: "https://accounts.google.com/o/oauth2/v2/auth",
        token_url: "https://oauth2.googleapis.com/token",
        scopes: "https://www.googleapis.com/auth/cloud-platform \
                 https://www.googleapis.com/auth/userinfo.email \
                 https://www.googleapis.com/auth/userinfo.profile",
        default_client_id: Some(GOOGLE_CLIENT_ID),
        default_client_secret: Some(GOOGLE_CLIENT_SECRET),
        redirect: RedirectStrategy::Localhost {
            path: REDIRECT_PATH,
            fixed_port: None, // dynamic port
        },
        extra_auth_params: &[("access_type", "offline"), ("prompt", "consent")],
        state_is_verifier: false,
    }
}

/// Authentication provider for Google Gemini.
///
/// Supports two discovery paths:
///
/// 1. If the `GOOGLE_API_KEY` environment variable is set, an API-key auth
///    method is returned.
/// 2. Otherwise (and always as a fallback), an OAuth browser flow is offered.
pub struct GoogleAuthProvider {
    provider_id: ProviderId,
    http: reqwest::Client,
    /// Held between `start_login` and `complete_login` so the verifier is
    /// available for the token exchange.
    pkce: std::sync::Mutex<Option<PkceChallenge>>,
    /// The port the local listener bound to, held so `complete_login` can
    /// reconstruct the redirect URI.
    bound_port: std::sync::Mutex<Option<u16>>,
}

impl std::fmt::Debug for GoogleAuthProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleAuthProvider")
            .field("provider_id", &self.provider_id)
            .finish_non_exhaustive()
    }
}

impl GoogleAuthProvider {
    pub fn new() -> Self {
        Self {
            provider_id: PROVIDER_ID.clone(),
            http: reqwest::Client::new(),
            pkce: std::sync::Mutex::new(None),
            bound_port: std::sync::Mutex::new(None),
        }
    }
}

impl Default for GoogleAuthProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl AuthProvider for GoogleAuthProvider {
    fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    async fn discover(&self) -> Result<Vec<AuthMethod>, FrameworkError> {
        let mut methods = Vec::new();

        // Check for an existing API key in the environment.
        if let Ok(key) = std::env::var("GOOGLE_API_KEY") {
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
        let endpoints = google_endpoints();
        let pkce = PkceChallenge::generate();
        let state = generate_state();

        // Use port 0 — the presentation layer will bind to a free port and
        // tell `complete_login` which port was used via the `redirect_uri` param.
        let redirect_uri = redirect_uri_for(&endpoints.redirect, 0);

        let url = build_auth_url(&endpoints, GOOGLE_CLIENT_ID, &redirect_uri, &pkce, &state)?;

        // Stash the PKCE verifier so `complete_login` can use it.
        *self.pkce.lock().unwrap() = Some(pkce);
        *self.bound_port.lock().unwrap() = Some(0);

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

        let endpoints = google_endpoints();

        // If the caller provided a redirect_uri (e.g. with the actual bound port),
        // prefer that. Otherwise reconstruct from default.
        let redirect_uri = params
            .get("redirect_uri")
            .cloned()
            .unwrap_or_else(|| redirect_uri_for(&endpoints.redirect, 0));

        // Google requires client_secret in the token exchange.
        let token_resp: OAuthTokenResponse = self
            .http
            .post(endpoints.token_url)
            .form(&[
                ("grant_type", "authorization_code"),
                ("code", code.as_str()),
                ("redirect_uri", redirect_uri.as_str()),
                ("client_id", GOOGLE_CLIENT_ID),
                ("client_secret", GOOGLE_CLIENT_SECRET),
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

        let endpoints = google_endpoints();

        // Google requires client_secret for refresh as well.
        let token_resp: OAuthTokenResponse = self
            .http
            .post(endpoints.token_url)
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token),
                ("client_id", GOOGLE_CLIENT_ID),
                ("client_secret", GOOGLE_CLIENT_SECRET),
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
