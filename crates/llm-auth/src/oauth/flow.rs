use llm_core::FrameworkError;
use url::Url;

use super::pkce::PkceChallenge;
use super::primitives::{OAuthEndpoints, OAuthTokenResponse, RedirectStrategy};

/// Async trait for performing an OAuth 2.0 authorization-code flow.
///
/// Provider crates supply concrete implementations; this crate only
/// defines the interface and the URL-construction helper.
#[async_trait::async_trait]
pub trait OAuthFlow: Send + Sync {
    /// Return the provider's endpoint configuration.
    fn endpoints(&self) -> &OAuthEndpoints;

    /// Build the authorization URL the user should open in a browser.
    ///
    /// The default implementation constructs a standard OAuth 2.0 + PKCE
    /// authorization URL using the endpoint configuration.
    fn build_auth_url(
        &self,
        client_id: &str,
        redirect_uri: &str,
        pkce: &PkceChallenge,
        state: &str,
    ) -> Result<String, FrameworkError> {
        build_auth_url(self.endpoints(), client_id, redirect_uri, pkce, state)
    }

    /// Exchange an authorization code for tokens.
    async fn exchange_code(
        &self,
        code: &str,
        redirect_uri: &str,
        pkce_verifier: &str,
        state: Option<&str>,
    ) -> Result<OAuthTokenResponse, FrameworkError>;

    /// Use a refresh token to obtain a fresh access token.
    async fn refresh_token(
        &self,
        refresh_token: &str,
    ) -> Result<OAuthTokenResponse, FrameworkError>;
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Construct the redirect URI for the given strategy and local port.
///
/// For `Localhost` strategies the URI is `http://localhost:{port}{path}`.
/// For `RemoteCallback` the fixed URI is returned as-is.
/// For `DeviceCode` and `CustomScheme` the caller should not need a
/// redirect URI, so an empty string is returned.
#[must_use]
pub fn redirect_uri_for(redirect: &RedirectStrategy, bound_port: u16) -> String {
    match redirect {
        RedirectStrategy::Localhost { path, .. } => {
            let suffix = path.trim_start_matches('/');
            if suffix.is_empty() {
                format!("http://localhost:{bound_port}")
            } else {
                format!("http://localhost:{bound_port}/{suffix}")
            }
        }
        RedirectStrategy::RemoteCallback { redirect_uri } => (*redirect_uri).to_string(),
        RedirectStrategy::DeviceCode => String::new(),
        RedirectStrategy::CustomScheme { scheme } => format!("{scheme}://callback"),
    }
}

/// Build a standard OAuth 2.0 + PKCE authorization URL.
pub fn build_auth_url(
    endpoints: &OAuthEndpoints,
    client_id: &str,
    redirect_uri: &str,
    pkce: &PkceChallenge,
    state: &str,
) -> Result<String, FrameworkError> {
    let mut url = Url::parse(endpoints.auth_url)
        .map_err(|e| FrameworkError::config(format!("invalid OAuth auth_url: {e}")))?;
    url.query_pairs_mut()
        .append_pair("client_id", client_id)
        .append_pair("redirect_uri", redirect_uri)
        .append_pair("response_type", "code")
        .append_pair("scope", endpoints.scopes)
        .append_pair("code_challenge_method", "S256")
        .append_pair("code_challenge", &pkce.challenge)
        .append_pair("state", state);
    for &(key, value) in endpoints.extra_auth_params {
        url.query_pairs_mut().append_pair(key, value);
    }
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oauth::primitives::RedirectStrategy;

    fn test_endpoints() -> OAuthEndpoints {
        OAuthEndpoints {
            auth_url: "https://example.com/oauth/authorize",
            token_url: "https://example.com/oauth/token",
            scopes: "read write",
            default_client_id: Some("test-client"),
            default_client_secret: None,
            redirect: RedirectStrategy::Localhost {
                path: "/callback",
                fixed_port: Some(8080),
            },
            extra_auth_params: &[("prompt", "consent")],
            state_is_verifier: false,
        }
    }

    #[test]
    fn build_auth_url_contains_required_params() {
        let endpoints = test_endpoints();
        let pkce = PkceChallenge::generate();
        let state = "test-state-abc";
        let redirect = "http://localhost:8080/callback";

        let url_str = build_auth_url(&endpoints, "my-client", redirect, &pkce, state)
            .expect("should build a valid URL");
        let url = Url::parse(&url_str).expect("should be a valid URL");

        assert_eq!(url.scheme(), "https");
        assert_eq!(url.host_str(), Some("example.com"));
        assert_eq!(url.path(), "/oauth/authorize");

        let pairs: std::collections::HashMap<_, _> = url.query_pairs().collect();
        assert_eq!(pairs.get("client_id").unwrap(), "my-client");
        assert_eq!(pairs.get("redirect_uri").unwrap(), redirect);
        assert_eq!(pairs.get("response_type").unwrap(), "code");
        assert_eq!(pairs.get("scope").unwrap(), "read write");
        assert_eq!(pairs.get("code_challenge_method").unwrap(), "S256");
        assert_eq!(pairs.get("code_challenge").unwrap(), &pkce.challenge);
        assert_eq!(pairs.get("state").unwrap(), state);
        // Extra params
        assert_eq!(pairs.get("prompt").unwrap(), "consent");
    }

    #[test]
    fn redirect_uri_for_localhost() {
        let strategy = RedirectStrategy::Localhost {
            path: "/oauth2callback",
            fixed_port: None,
        };
        let uri = redirect_uri_for(&strategy, 9999);
        assert_eq!(uri, "http://localhost:9999/oauth2callback");
    }

    #[test]
    fn redirect_uri_for_localhost_empty_path() {
        let strategy = RedirectStrategy::Localhost {
            path: "/",
            fixed_port: None,
        };
        let uri = redirect_uri_for(&strategy, 3000);
        assert_eq!(uri, "http://localhost:3000");
    }

    #[test]
    fn redirect_uri_for_remote() {
        let strategy = RedirectStrategy::RemoteCallback {
            redirect_uri: "https://example.com/callback",
        };
        let uri = redirect_uri_for(&strategy, 0);
        assert_eq!(uri, "https://example.com/callback");
    }

    #[test]
    fn redirect_uri_for_custom_scheme() {
        let strategy = RedirectStrategy::CustomScheme { scheme: "myapp" };
        let uri = redirect_uri_for(&strategy, 0);
        assert_eq!(uri, "myapp://callback");
    }

    #[test]
    fn redirect_uri_for_device_code_is_empty() {
        let strategy = RedirectStrategy::DeviceCode;
        let uri = redirect_uri_for(&strategy, 0);
        assert!(uri.is_empty());
    }
}
