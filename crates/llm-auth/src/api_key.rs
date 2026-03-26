use llm_core::{FrameworkError, ProviderId};

/// Trait for a backing store that can persist and retrieve API keys.
///
/// The `ApiKeyResolver` checks the environment first, then falls back
/// to this store. Provider crates or the app layer supply the concrete
/// implementation (e.g. OS keychain, encrypted file, etc.).
#[async_trait::async_trait]
pub trait ApiKeyStore: Send + Sync {
    /// Look up a stored API key for the given provider.
    async fn get(&self, provider: &ProviderId) -> Result<Option<String>, FrameworkError>;

    /// Persist an API key for the given provider.
    async fn set(&self, provider: &ProviderId, key: &str) -> Result<(), FrameworkError>;

    /// Remove a stored API key for the given provider.
    async fn remove(&self, provider: &ProviderId) -> Result<(), FrameworkError>;
}

/// Resolves an API key for a provider by checking, in order:
///
/// 1. A specific environment variable (e.g. `OPENAI_API_KEY`).
/// 2. A pluggable backing store (keychain, config file, etc.).
pub struct ApiKeyResolver<S: ApiKeyStore> {
    store: S,
}

impl<S: ApiKeyStore> ApiKeyResolver<S> {
    pub fn new(store: S) -> Self {
        Self { store }
    }

    /// Resolve an API key for `provider`.
    ///
    /// `env_var` is the name of the environment variable to check first
    /// (e.g. `"OPENAI_API_KEY"`). If the env var is set and non-empty its
    /// value is returned. Otherwise the backing store is queried.
    pub async fn resolve(
        &self,
        provider: &ProviderId,
        env_var: &str,
    ) -> Result<Option<String>, FrameworkError> {
        // 1. Check environment variable.
        if let Ok(value) = std::env::var(env_var) {
            if !value.is_empty() {
                return Ok(Some(value));
            }
        }

        // 2. Fall back to the backing store.
        self.store.get(provider).await
    }

    /// Convenience: resolve, returning an `Err` if no key is found.
    pub async fn resolve_required(
        &self,
        provider: &ProviderId,
        env_var: &str,
    ) -> Result<String, FrameworkError> {
        self.resolve(provider, env_var).await?.ok_or_else(|| {
            FrameworkError::auth(format!(
                "no API key found for provider {provider} \
                 (checked ${env_var} and key store)"
            ))
        })
    }

    /// Expose the underlying store for direct access.
    pub fn store(&self) -> &S {
        &self.store
    }
}
