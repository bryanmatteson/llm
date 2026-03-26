use chrono::{DateTime, Utc};
use llm_core::{FrameworkError, Metadata, ProviderId};
use serde::{Deserialize, Serialize};

use crate::token::TokenPair;

// ── AuthMethod ───────────────────────────────────────────────────────

/// The concrete authentication mechanism in use for a provider session.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AuthMethod {
    /// A static API key (e.g. `sk-...`).
    ApiKey {
        masked: String,
    },
    /// A bearer token without an associated OAuth flow.
    Bearer {
        expires_at: Option<DateTime<Utc>>,
    },
    /// Full OAuth 2.0 session with token pair.
    OAuth {
        expires_at: DateTime<Utc>,
    },
}

// ── AuthSession ──────────────────────────────────────────────────────

/// Represents a fully-authenticated session with a provider.
///
/// Returned by `AuthProvider::complete_login` and persisted between
/// application launches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthSession {
    pub provider_id: ProviderId,
    pub method: AuthMethod,
    pub tokens: TokenPair,
    #[serde(default)]
    pub metadata: Metadata,
}

// ── AuthStart ────────────────────────────────────────────────────────

/// Describes what the UI/CLI should do after `AuthProvider::start_login`.
///
/// The auth crate never performs terminal I/O; it returns one of these
/// variants and the presentation layer decides how to present it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthStart {
    /// The user must open `url` in a browser. The auth layer is
    /// listening for the callback on `redirect_uri`.
    OAuthBrowser {
        url: String,
        redirect_uri: String,
        /// Opaque state the provider needs to complete login.
        state: String,
    },
    /// The user must enter a device code at `verification_uri`.
    DeviceCode {
        verification_uri: String,
        user_code: String,
        /// How often (in seconds) the caller should poll for completion.
        interval: u64,
    },
    /// The user must supply an API key.
    ApiKeyPrompt {
        env_var_hint: String,
    },
}

// ── AuthCompletion ───────────────────────────────────────────────────

/// The result of successfully completing a login flow.
#[derive(Debug, Clone)]
pub struct AuthCompletion {
    pub session: AuthSession,
}

// ── AuthProvider trait ───────────────────────────────────────────────

/// The core authentication trait that every provider must implement.
///
/// The flow is split into discrete steps so the auth crate stays free
/// of terminal I/O:
///
/// 1. `discover()` – figure out what auth methods are available.
/// 2. `start_login()` – begin a login flow, returning an [`AuthStart`]
///    that tells the UI what to present.
/// 3. `complete_login()` – finish the flow once the user has acted
///    (e.g. paste the callback code, enter an API key).
/// 4. `refresh()` / `validate()` / `logout()` for session management.
#[async_trait::async_trait]
pub trait AuthProvider: Send + Sync {
    /// Return the provider identifier this implementation handles.
    fn provider_id(&self) -> &ProviderId;

    /// Discover available authentication methods.
    ///
    /// For example, a provider might support both OAuth and API-key auth.
    /// Returns a list of [`AuthMethod`] variants (without credential data)
    /// to let the UI offer a choice.
    async fn discover(&self) -> Result<Vec<AuthMethod>, FrameworkError>;

    /// Begin a login flow.
    ///
    /// Returns an [`AuthStart`] that tells the presentation layer what
    /// to show the user (browser URL, device code, API-key prompt, etc.).
    async fn start_login(&self) -> Result<AuthStart, FrameworkError>;

    /// Complete a login flow.
    ///
    /// `params` is a freeform map whose keys depend on the flow:
    ///   - OAuth browser: `{ "code": "...", "state": "..." }`
    ///   - Device code: may be empty (the impl polls internally)
    ///   - API key prompt: `{ "api_key": "sk-..." }`
    async fn complete_login(
        &self,
        params: &Metadata,
    ) -> Result<AuthCompletion, FrameworkError>;

    /// End the current session and clean up any stored tokens.
    async fn logout(&self, session: &AuthSession) -> Result<(), FrameworkError>;

    /// Refresh an existing session (typically by using a refresh token).
    ///
    /// Returns a new `AuthSession` with fresh tokens.
    async fn refresh(&self, session: &AuthSession) -> Result<AuthSession, FrameworkError>;

    /// Validate that an existing session is still usable.
    ///
    /// Returns `true` if the session's token is valid and not expired.
    async fn validate(&self, session: &AuthSession) -> Result<bool, FrameworkError>;
}
