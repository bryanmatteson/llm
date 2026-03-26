use llm_core::{Metadata, ProviderId};

use crate::provider::{AuthMethod, AuthSession};
use crate::token::TokenPair;

// ── Types ──────────────────────────────────────────────────────────────

/// How the credential was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialKind {
    /// A static API key (e.g. `OPENAI_API_KEY`).
    ApiKey,
    /// A bearer / OAuth access token.
    BearerToken,
}

/// A credential discovered from the environment.
#[derive(Debug, Clone)]
pub struct ProviderCredential {
    pub token: String,
    pub kind: CredentialKind,
    pub base_url: Option<String>,
}

/// Describes which environment variables to probe for a single provider.
#[derive(Debug, Clone)]
pub struct ProviderEnvConfig {
    /// Canonical provider name (e.g. `"openai"`).
    pub provider: &'static str,
    /// Alternative names that should resolve to this provider
    /// (e.g. `["codex"]` for OpenAI).
    pub aliases: &'static [&'static str],
    /// Env vars to check for API key auth, in priority order.
    pub api_key_vars: &'static [&'static str],
    /// Env vars to check for bearer / OAuth token, in priority order.
    pub access_token_vars: &'static [&'static str],
    /// Env vars to check for a custom base URL override.
    pub base_url_vars: &'static [&'static str],
}

// ── Default configs for well-known providers ───────────────────────────

const OPENAI_CONFIG: ProviderEnvConfig = ProviderEnvConfig {
    provider: "openai",
    aliases: &["codex"],
    api_key_vars: &["OPENAI_API_KEY"],
    access_token_vars: &["OPENAI_ACCESS_TOKEN"],
    base_url_vars: &["OPENAI_BASE_URL"],
};

const ANTHROPIC_CONFIG: ProviderEnvConfig = ProviderEnvConfig {
    provider: "anthropic",
    aliases: &["claude"],
    api_key_vars: &["ANTHROPIC_API_KEY"],
    access_token_vars: &["ANTHROPIC_ACCESS_TOKEN"],
    base_url_vars: &[],
};

const GOOGLE_CONFIG: ProviderEnvConfig = ProviderEnvConfig {
    provider: "google",
    aliases: &["gemini"],
    api_key_vars: &["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    access_token_vars: &[
        "GOOGLE_ACCESS_TOKEN",
        "GOOGLE_OAUTH_ACCESS_TOKEN",
        "GEMINI_ACCESS_TOKEN",
    ],
    base_url_vars: &[],
};

// ── Discovery ──────────────────────────────────────────────────────────

/// Discovers LLM provider credentials from environment variables.
///
/// Supports an optional prefix (e.g. `"STAG_"`) that is checked first,
/// falling back to the unprefixed variable names.
pub struct EnvCredentialDiscovery {
    configs: Vec<ProviderEnvConfig>,
    prefix: Option<String>,
}

impl EnvCredentialDiscovery {
    /// Create a discovery instance with the three default providers
    /// (OpenAI, Anthropic, Google) and no env-var prefix.
    pub fn with_defaults() -> Self {
        Self {
            configs: vec![
                OPENAI_CONFIG.clone(),
                ANTHROPIC_CONFIG.clone(),
                GOOGLE_CONFIG.clone(),
            ],
            prefix: None,
        }
    }

    /// Create a discovery instance with a prefix.
    ///
    /// For each env var `FOO_API_KEY`, the prefixed form `{prefix}FOO_API_KEY`
    /// is checked first. If not found, `FOO_API_KEY` is checked as a fallback.
    ///
    /// Example: `EnvCredentialDiscovery::with_prefix("STAG_")` checks
    /// `STAG_OPENAI_API_KEY` before `OPENAI_API_KEY`.
    pub fn with_prefix(prefix: impl Into<String>) -> Self {
        Self {
            configs: vec![
                OPENAI_CONFIG.clone(),
                ANTHROPIC_CONFIG.clone(),
                GOOGLE_CONFIG.clone(),
            ],
            prefix: Some(prefix.into()),
        }
    }

    /// Add a custom provider config.
    pub fn add_provider(&mut self, config: ProviderEnvConfig) {
        self.configs.push(config);
    }

    /// Discover credentials for a specific provider by canonical name or alias.
    ///
    /// Returns `None` if no credential environment variable is set.
    pub fn discover(&self, provider: &str) -> Option<ProviderCredential> {
        let normalized = provider.trim().to_ascii_lowercase();
        let config = self.configs.iter().find(|c| {
            c.provider == normalized || c.aliases.iter().any(|a| *a == normalized)
        })?;

        let base_url = self.first_env(config.base_url_vars);

        if let Some(token) = self.first_env(config.api_key_vars) {
            return Some(ProviderCredential {
                token,
                kind: CredentialKind::ApiKey,
                base_url,
            });
        }

        if let Some(token) = self.first_env(config.access_token_vars) {
            return Some(ProviderCredential {
                token,
                kind: CredentialKind::BearerToken,
                base_url,
            });
        }

        None
    }

    /// Try each provider in `priority` order and return the first one that
    /// has credentials.  Returns `(canonical_provider_name, credential)`.
    pub fn discover_any(&self, priority: &[&str]) -> Option<(String, ProviderCredential)> {
        for provider in priority {
            if let Some(cred) = self.discover(provider) {
                let canonical = self
                    .configs
                    .iter()
                    .find(|c| {
                        c.provider == *provider || c.aliases.iter().any(|a| *a == *provider)
                    })
                    .map(|c| c.provider)
                    .unwrap_or(provider);
                return Some((canonical.to_string(), cred));
            }
        }
        None
    }

    /// Check env vars with optional prefix fallback.
    fn first_env(&self, vars: &[&str]) -> Option<String> {
        // If we have a prefix, check prefixed versions first.
        if let Some(prefix) = &self.prefix {
            for var in vars {
                let prefixed = format!("{prefix}{var}");
                if let Some(val) = read_env(&prefixed) {
                    return Some(val);
                }
            }
        }
        // Then check unprefixed.
        for var in vars {
            if let Some(val) = read_env(var) {
                return Some(val);
            }
        }
        None
    }
}

/// Build an [`AuthSession`] from a discovered credential.
///
/// API keys get a long-lived (365-day) token pair.
/// Bearer tokens get a shorter-lived (1-hour) token pair.
pub fn build_auth_session(
    provider_id: ProviderId,
    credential: &ProviderCredential,
) -> AuthSession {
    let (method, expires_secs) = match credential.kind {
        CredentialKind::ApiKey => {
            let masked = mask_token(&credential.token);
            (AuthMethod::ApiKey { masked }, 365 * 24 * 3600)
        }
        CredentialKind::BearerToken => (
            AuthMethod::Bearer {
                expires_at: None,
            },
            3600,
        ),
    };

    AuthSession {
        provider_id,
        method,
        tokens: TokenPair::new(credential.token.clone(), None, expires_secs),
        metadata: Metadata::default(),
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn read_env(var: &str) -> Option<String> {
    match std::env::var(var) {
        Ok(val) => {
            let trimmed = val.trim().to_string();
            if trimmed.is_empty() { None } else { Some(trimmed) }
        }
        Err(_) => None,
    }
}

fn mask_token(token: &str) -> String {
    if token.len() <= 8 {
        "****".to_string()
    } else {
        format!("{}…{}", &token[..4], &token[token.len() - 4..])
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_returns_none_when_no_env_set() {
        // Use a provider with unusual env vars that are unlikely to be set.
        let mut discovery = EnvCredentialDiscovery::with_defaults();
        discovery.add_provider(ProviderEnvConfig {
            provider: "test_provider_xyz",
            aliases: &[],
            api_key_vars: &["LLM_TEST_XYZ_API_KEY_NONEXISTENT"],
            access_token_vars: &["LLM_TEST_XYZ_TOKEN_NONEXISTENT"],
            base_url_vars: &[],
        });
        assert!(discovery.discover("test_provider_xyz").is_none());
    }

    #[test]
    fn alias_resolution_works() {
        let discovery = EnvCredentialDiscovery::with_defaults();
        // "codex" should find the openai config (even if no env is set)
        let config = discovery
            .configs
            .iter()
            .find(|c| c.provider == "openai" || c.aliases.iter().any(|a| *a == "codex"));
        assert!(config.is_some());
        assert_eq!(config.unwrap().provider, "openai");
    }

    #[test]
    fn mask_token_short() {
        assert_eq!(mask_token("abc"), "****");
    }

    #[test]
    fn mask_token_long() {
        assert_eq!(mask_token("sk-1234567890abcdef"), "sk-1…cdef");
    }

    #[test]
    fn build_auth_session_api_key() {
        let cred = ProviderCredential {
            token: "sk-test-key-12345678".to_string(),
            kind: CredentialKind::ApiKey,
            base_url: None,
        };
        let session = build_auth_session(ProviderId::new("openai"), &cred);
        assert_eq!(session.provider_id.as_str(), "openai");
        assert!(matches!(session.method, AuthMethod::ApiKey { .. }));
        assert!(!session.tokens.is_expired());
    }

    #[test]
    fn build_auth_session_bearer() {
        let cred = ProviderCredential {
            token: "ya29.bearer-token".to_string(),
            kind: CredentialKind::BearerToken,
            base_url: Some("https://custom.api.example.com".to_string()),
        };
        let session = build_auth_session(ProviderId::new("google"), &cred);
        assert!(matches!(session.method, AuthMethod::Bearer { .. }));
        assert!(!session.tokens.is_expired());
    }

    #[test]
    fn discover_any_returns_none_when_empty_priority() {
        let discovery = EnvCredentialDiscovery::with_defaults();
        assert!(discovery.discover_any(&[]).is_none());
    }
}
