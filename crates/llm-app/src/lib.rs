pub mod auth_service;
pub mod builder;
pub mod context;
pub mod questionnaire_service;
pub mod registry;
pub mod session_service;
pub mod skill_service;
pub mod tool_service;

// ── Re-exports ─────────────────────────────────────────────────────

pub use auth_service::AuthService;
pub use builder::AppBuilder;
pub use context::LlmContext;
pub use questionnaire_service::QuestionnaireService;
pub use registry::{ProviderClientFactory, ProviderRegistration, ProviderRegistry};
pub use session_service::SessionService;
pub use skill_service::SkillService;
pub use tool_service::ToolService;

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::pin::Pin;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

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

    fn tempdir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("llm-app-{prefix}-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_skill_dir(parent: &Path, name: &str, body: &str) -> PathBuf {
        let dir = parent.join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {body}\n---\n\n# {name}\n"),
        )
        .unwrap();
        dir
    }

    fn stub_auth_session(provider: &str) -> AuthSession {
        AuthSession {
            provider_id: ProviderId::new(provider),
            method: AuthMethod::ApiKey {
                masked: "sk-****".into(),
            },
            tokens: TokenPair::new("tok".into(), None, 3600),
            metadata: BTreeMap::new(),
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

    #[test]
    fn build_fails_when_skill_loading_fails() {
        let root = tempdir("bad-skill");
        let bad_dir = root.join("broken");
        fs::create_dir_all(&bad_dir).unwrap();
        fs::write(
            bad_dir.join("SKILL.md"),
            "---\nname: BAD_NAME\ndescription: Broken skill.\n---\n",
        )
        .unwrap();

        let err = AppBuilder::new().with_skill_dir(&root).build().unwrap_err();
        assert!(matches!(err, FrameworkError::Config { .. }));
        assert!(
            err.to_string()
                .contains("failed to load one or more skills")
        );
        assert!(err.to_string().contains("broken"));
    }

    #[tokio::test]
    async fn create_session_injects_registered_skill_metadata() {
        let root = tempdir("skills");
        let system_dir = root.join(".system");
        make_skill_dir(&system_dir, "openai-docs", "Use official OpenAI docs.");

        let ctx = AppBuilder::new()
            .with_skill_dir(&root)
            .register_provider(make_registration("alpha"))
            .build()
            .unwrap();

        let config = llm_core::SessionConfig::for_provider("alpha");
        let auth = stub_auth_session("alpha");
        let (handle, _tx, _rx) = ctx
            .sessions
            .create_session(&ProviderId::new("alpha"), &auth, config)
            .await
            .unwrap();

        let prompt = handle.config.system_prompt.as_deref().unwrap();
        assert!(prompt.contains("The following skills are available:"));
        assert!(prompt.contains("- openai-docs: Use official OpenAI docs."));
    }

    #[tokio::test]
    async fn create_session_preserves_existing_system_prompt() {
        let root = tempdir("prompt-merge");
        make_skill_dir(&root, "pdf", "Extract text from PDFs.");

        let ctx = AppBuilder::new()
            .with_skill_dir(&root)
            .register_provider(make_registration("alpha"))
            .build()
            .unwrap();

        let mut config = llm_core::SessionConfig::for_provider("alpha");
        config.system_prompt = Some("Be concise.".into());

        let auth = stub_auth_session("alpha");
        let (handle, _tx, _rx) = ctx
            .sessions
            .create_session(&ProviderId::new("alpha"), &auth, config)
            .await
            .unwrap();

        let prompt = handle.config.system_prompt.as_deref().unwrap();
        assert!(prompt.starts_with("Be concise."));
        assert!(prompt.contains("The following skills are available:"));
        assert!(prompt.contains("- pdf: Extract text from PDFs."));
    }
}
