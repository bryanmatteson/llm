use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use llm_auth::{AuthProvider, AuthSession};
use llm_core::{ModelId, ProviderId, ProviderDescriptor, Result};
use llm_provider_api::{LlmProviderClient, ToolSchemaAdapter};

// ── ProviderClientFactory ──────────────────────────────────────────

/// Factory trait for creating provider-specific LLM clients.
///
/// Given an authenticated session and a target model, the factory produces
/// a boxed [`LlmProviderClient`] ready to send turns.
#[async_trait]
pub trait ProviderClientFactory: Send + Sync {
    /// Create a new provider client for the given auth session and model.
    async fn create_client(
        &self,
        auth: &AuthSession,
        model: &ModelId,
    ) -> Result<Box<dyn LlmProviderClient>>;
}

// ── ProviderRegistration ───────────────────────────────────────────

/// A complete registration bundle for a single LLM provider.
///
/// Groups the provider's static descriptor with the runtime components
/// needed to authenticate, create clients, and translate tool schemas.
pub struct ProviderRegistration {
    /// Static descriptor (id, display name, default model, capabilities).
    pub descriptor: ProviderDescriptor,
    /// The authentication provider implementation for this provider.
    pub auth_provider: Arc<dyn AuthProvider>,
    /// Factory for creating LLM clients once authenticated.
    pub client_factory: Arc<dyn ProviderClientFactory>,
    /// Adapter for translating tool schemas into the provider's wire format.
    pub tool_adapter: Arc<dyn ToolSchemaAdapter>,
}

// ── ProviderRegistry ───────────────────────────────────────────────

/// Central registry of all available LLM providers.
///
/// The registry maps [`ProviderId`]s to their [`ProviderRegistration`] bundles,
/// making it easy to look up any component by provider id.
pub struct ProviderRegistry {
    providers: HashMap<ProviderId, ProviderRegistration>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a provider. If a provider with the same id already exists it
    /// will be replaced.
    pub fn register(&mut self, registration: ProviderRegistration) {
        let id = registration.descriptor.id.clone();
        self.providers.insert(id, registration);
    }

    /// Look up a provider registration by its id.
    pub fn get(&self, id: &ProviderId) -> Option<&ProviderRegistration> {
        self.providers.get(id)
    }

    /// Return the descriptors for every registered provider, sorted by
    /// provider id for deterministic output.
    pub fn list_providers(&self) -> Vec<&ProviderDescriptor> {
        let mut descriptors: Vec<&ProviderDescriptor> = self
            .providers
            .values()
            .map(|r| &r.descriptor)
            .collect();
        descriptors.sort_by(|a, b| a.id.cmp(&b.id));
        descriptors
    }

    /// Return the ids of every registered provider, sorted.
    pub fn list_provider_ids(&self) -> Vec<ProviderId> {
        let mut ids: Vec<ProviderId> = self.providers.keys().cloned().collect();
        ids.sort();
        ids
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}
