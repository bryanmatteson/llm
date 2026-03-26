use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::{Digest, Sha256};

/// A PKCE (Proof Key for Code Exchange) challenge pair.
///
/// The `verifier` is kept secret and sent during the token exchange.
/// The `challenge` (a SHA-256 hash of the verifier) is sent in the
/// authorization URL so the provider can verify them later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PkceChallenge {
    pub verifier: String,
    pub challenge: String,
}

impl PkceChallenge {
    /// Generate a new random PKCE challenge.
    ///
    /// Uses 32 cryptographically-suitable random bytes, base64url-encodes them
    /// as the verifier, then SHA-256 hashes and base64url-encodes the result
    /// as the challenge.
    #[must_use]
    pub fn generate() -> Self {
        let mut bytes = [0_u8; 32];
        getrandom::fill(&mut bytes).expect("failed to generate random bytes");
        let verifier = URL_SAFE_NO_PAD.encode(bytes);
        let digest = Sha256::digest(verifier.as_bytes());
        let challenge = URL_SAFE_NO_PAD.encode(digest);
        Self {
            verifier,
            challenge,
        }
    }
}

/// Generate a random opaque state parameter for OAuth requests.
///
/// Returns 16 random bytes encoded as a URL-safe base64 string (no padding).
#[must_use]
pub fn generate_state() -> String {
    let mut bytes = [0_u8; 16];
    getrandom::fill(&mut bytes).expect("failed to generate random bytes");
    URL_SAFE_NO_PAD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use sha2::{Digest, Sha256};

    use super::*;

    #[test]
    fn pkce_verifier_is_base64url() {
        let pkce = PkceChallenge::generate();
        // The verifier should be decodable from URL-safe base64.
        let decoded = URL_SAFE_NO_PAD.decode(&pkce.verifier);
        assert!(decoded.is_ok(), "verifier should be valid base64url");
        assert_eq!(decoded.unwrap().len(), 32, "verifier should decode to 32 bytes");
    }

    #[test]
    fn pkce_challenge_matches_verifier() {
        let pkce = PkceChallenge::generate();
        // Recompute the challenge from the verifier.
        let digest = Sha256::digest(pkce.verifier.as_bytes());
        let expected_challenge = URL_SAFE_NO_PAD.encode(digest);
        assert_eq!(pkce.challenge, expected_challenge);
    }

    #[test]
    fn pkce_pairs_are_unique() {
        let a = PkceChallenge::generate();
        let b = PkceChallenge::generate();
        assert_ne!(a.verifier, b.verifier);
        assert_ne!(a.challenge, b.challenge);
    }

    #[test]
    fn generate_state_is_base64url() {
        let state = generate_state();
        let decoded = URL_SAFE_NO_PAD.decode(&state);
        assert!(decoded.is_ok(), "state should be valid base64url");
        assert_eq!(decoded.unwrap().len(), 16, "state should decode to 16 bytes");
    }

    #[test]
    fn generate_state_is_unique() {
        let a = generate_state();
        let b = generate_state();
        assert_ne!(a, b);
    }
}
