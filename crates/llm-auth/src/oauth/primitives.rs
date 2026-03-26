use serde::{Deserialize, Serialize};

/// OAuth provider endpoint configuration.
///
/// Describes the URLs, scopes, client credentials, and redirect strategy
/// needed to perform an OAuth 2.0 authorization-code flow with PKCE.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthEndpoints {
    pub auth_url: &'static str,
    pub token_url: &'static str,
    pub scopes: &'static str,
    pub default_client_id: Option<&'static str>,
    pub default_client_secret: Option<&'static str>,
    pub redirect: RedirectStrategy,
    pub extra_auth_params: &'static [(&'static str, &'static str)],
    /// When `true` the OAuth `state` parameter doubles as the PKCE verifier
    /// (used by some providers like Anthropic).
    pub state_is_verifier: bool,
}

/// How the OAuth callback is received after the user authorizes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RedirectStrategy {
    /// Spin up a temporary HTTP server on localhost.
    Localhost {
        path: &'static str,
        fixed_port: Option<u16>,
    },
    /// The provider redirects to a well-known remote URI that relays the code.
    RemoteCallback { redirect_uri: &'static str },
    /// The provider uses the device-code grant (no browser redirect).
    DeviceCode,
    /// A custom URI scheme registered by the application (e.g. `myapp://`).
    CustomScheme { scheme: &'static str },
}

/// The token response returned by an OAuth token endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuthTokenResponse {
    pub access_token: String,

    #[serde(default)]
    pub refresh_token: Option<String>,

    #[serde(default)]
    pub expires_in: Option<u64>,

    #[serde(default)]
    pub token_type: Option<String>,
}
