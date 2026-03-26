pub mod auth_service;
pub mod builder;
pub mod context;
pub mod questionnaire_service;
pub mod registry;
pub mod session_service;
pub mod tool_service;

// ── Re-exports ─────────────────────────────────────────────────────

pub use auth_service::AuthService;
pub use builder::AppBuilder;
pub use context::LlmContext;
pub use questionnaire_service::QuestionnaireService;
pub use registry::{ProviderClientFactory, ProviderRegistration, ProviderRegistry};
pub use session_service::SessionService;
pub use tool_service::ToolService;

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::pin::Pin;
    use std::sync::Arc;

    use async_trait::async_trait;
    use tokio_stream::Stream;

    use llm_auth::{AuthCompletion, AuthMethod, AuthProvider, AuthSession, AuthStart, TokenPair};
    use llm_core::{
        FrameworkError, Metadata, ModelDescriptor, ModelId, ProviderCapability, ProviderDescriptor,
        ProviderId, Result,
    };
    use llm_provider_api::{
        LlmProviderClient, ProviderEvent, ProviderToolCall, ProviderToolDescriptor,
        ToolSchemaAdapter, TurnRequest, TurnResponse,
    };

    use crate::builder::AppBuilder;
    use crate::registry::{ProviderClientFactory, ProviderRegistration};

    // -- Stub AuthProvider ---------------------------------------------------

    #[derive(Debug)]
    struct StubAuthProvider {
        id: ProviderId,
    }

    #[async_trait]
    impl AuthProvider for StubAuthProvider {
        fn provider_id(&self) -> &ProviderId {
            &self.id
        }

        async fn discover(&self) -> Result<Vec<AuthMethod>> {
            Ok(vec![AuthMethod::ApiKey {
                masked: "sk-****".into(),
            }])
        }

        async fn start_login(&self) -> Result<AuthStart> {
            Ok(AuthStart::ApiKeyPrompt {
                env_var_hint: "STUB_API_KEY".into(),
            })
        }

        async fn complete_login(&self, _params: &Metadata) -> Result<AuthCompletion> {
            Ok(AuthCompletion {
                session: AuthSession {
                    provider_id: self.id.clone(),
                    method: AuthMethod::ApiKey {
                        masked: "sk-****".into(),
                    },
                    tokens: TokenPair::new("tok".into(), None, 3600),
                    metadata: BTreeMap::new(),
                },
            })
        }

        async fn logout(&self, _session: &AuthSession) -> Result<()> {
            Ok(())
        }

        async fn refresh(&self, session: &AuthSession) -> Result<AuthSession> {
            Ok(session.clone())
        }

        async fn validate(&self, _session: &AuthSession) -> Result<bool> {
            Ok(true)
        }
    }

    // -- Stub ProviderClientFactory ------------------------------------------

    #[derive(Debug)]
    struct StubClientFactory {
        provider_id: ProviderId,
    }

    #[derive(Debug)]
    struct StubClient {
        provider_id: ProviderId,
    }

    #[async_trait]
    impl LlmProviderClient for StubClient {
        fn provider_id(&self) -> &ProviderId {
            &self.provider_id
        }

        async fn send_turn(&self, _request: &TurnRequest) -> Result<TurnResponse> {
            Err(FrameworkError::unsupported("stub"))
        }

        async fn stream_turn(
            &self,
            _request: &TurnRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>> {
            Err(FrameworkError::unsupported("stub"))
        }

        async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
            Ok(vec![])
        }
    }

    #[async_trait]
    impl ProviderClientFactory for StubClientFactory {
        async fn create_client(
            &self,
            _auth: &AuthSession,
            _model: &ModelId,
        ) -> Result<Box<dyn LlmProviderClient>> {
            Ok(Box::new(StubClient {
                provider_id: self.provider_id.clone(),
            }))
        }
    }

    // -- Stub ToolSchemaAdapter ----------------------------------------------

    struct StubAdapter;

    impl ToolSchemaAdapter for StubAdapter {
        fn translate_descriptors(
            &self,
            tools: &[ProviderToolDescriptor],
        ) -> Vec<serde_json::Value> {
            tools
                .iter()
                .map(|t| serde_json::to_value(t).unwrap())
                .collect()
        }

        fn parse_tool_calls(&self, _response: &serde_json::Value) -> Vec<ProviderToolCall> {
            vec![]
        }
    }

    // -- Helper: build a registration ----------------------------------------

    fn make_registration(name: &str) -> ProviderRegistration {
        let id = ProviderId::new(name);
        ProviderRegistration {
            descriptor: ProviderDescriptor {
                id: id.clone(),
                display_name: name.to_owned(),
                default_model: ModelId::new(format!("{name}-default")),
                capabilities: vec![ProviderCapability::ApiKeyAuth],
                metadata: BTreeMap::new(),
            },
            auth_provider: Arc::new(StubAuthProvider { id: id.clone() }),
            client_factory: Arc::new(StubClientFactory {
                provider_id: id.clone(),
            }),
            tool_adapter: Arc::new(StubAdapter),
        }
    }

    // -- Tests ---------------------------------------------------------------

    #[test]
    fn build_app_context_with_in_memory_stores() {
        let ctx = AppBuilder::new()
            .register_provider(make_registration("alpha"))
            .register_provider(make_registration("beta"))
            .build()
            .expect("build should succeed");

        let ids = ctx.providers.list_provider_ids();
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0].as_str(), "alpha");
        assert_eq!(ids[1].as_str(), "beta");

        let descriptors = ctx.providers.list_providers();
        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].display_name, "alpha");
        assert_eq!(descriptors[1].display_name, "beta");
    }

    #[test]
    fn build_empty_context() {
        let ctx = AppBuilder::new()
            .build()
            .expect("build with no providers should succeed");

        assert!(ctx.providers.list_provider_ids().is_empty());
        assert!(ctx.tools.list_tools().is_empty());
    }
}
