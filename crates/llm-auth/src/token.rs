use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// A pair of access + optional refresh token with an expiration timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenPair {
    pub access_token: String,
    #[serde(default)]
    pub refresh_token: Option<String>,
    pub expires_at: DateTime<Utc>,
}

impl TokenPair {
    /// Create a new token pair that expires `expires_in` seconds from now.
    #[must_use]
    pub fn new(access_token: String, refresh_token: Option<String>, expires_in_secs: i64) -> Self {
        let expires_at = Utc::now() + Duration::seconds(expires_in_secs);
        Self {
            access_token,
            refresh_token,
            expires_at,
        }
    }

    /// Returns `true` if the access token has already expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        Utc::now() >= self.expires_at
    }

    /// Returns `true` if the token will expire within the given buffer
    /// duration, making it a good time to refresh proactively.
    ///
    /// A typical buffer is 5 minutes.
    #[must_use]
    pub fn needs_refresh(&self, buffer: Duration) -> bool {
        Utc::now() + buffer >= self.expires_at
    }

    /// Returns `true` if a refresh token is available.
    #[must_use]
    pub fn can_refresh(&self) -> bool {
        self.refresh_token.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_token_is_not_expired() {
        let token = TokenPair::new("access".into(), None, 3600);
        assert!(!token.is_expired());
    }

    #[test]
    fn expired_token_is_detected() {
        let token = TokenPair {
            access_token: "access".into(),
            refresh_token: None,
            expires_at: Utc::now() - Duration::seconds(10),
        };
        assert!(token.is_expired());
    }

    #[test]
    fn needs_refresh_with_buffer() {
        // Token expires in 2 minutes — a 5-minute buffer should trigger refresh.
        let token = TokenPair::new("access".into(), Some("refresh".into()), 120);
        assert!(token.needs_refresh(Duration::minutes(5)));
        // But a 1-minute buffer should not.
        assert!(!token.needs_refresh(Duration::minutes(1)));
    }

    #[test]
    fn can_refresh_requires_refresh_token() {
        let with = TokenPair::new("a".into(), Some("r".into()), 3600);
        let without = TokenPair::new("a".into(), None, 3600);
        assert!(with.can_refresh());
        assert!(!without.can_refresh());
    }
}
