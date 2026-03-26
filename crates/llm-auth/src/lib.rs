pub mod api_key;
pub mod oauth;
pub mod provider;
pub mod token;

// ── Re-exports ───────────────────────────────────────────────────────

pub use api_key::{ApiKeyResolver, ApiKeyStore};
pub use oauth::{
    OAuthEndpoints, OAuthFlow, OAuthTokenResponse, PkceChallenge, RedirectStrategy, build_auth_url,
    generate_state, redirect_uri_for,
};
pub use provider::{AuthCompletion, AuthMethod, AuthProvider, AuthSession, AuthStart};
pub use token::TokenPair;
