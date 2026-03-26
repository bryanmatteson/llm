use llm_core::{ProviderId, Result};
use llm_questionnaire::{AnswerMap, QuestionId, Questionnaire, QuestionnaireRun};
use llm_session::SessionConfig;
use llm_tools::ToolPolicy;

/// Service for managing questionnaire flows and converting answers into
/// session configuration.
pub struct QuestionnaireService;

impl QuestionnaireService {
    /// Create a new `QuestionnaireService`.
    pub fn new() -> Self {
        Self
    }

    /// Start a questionnaire run from the given questionnaire definition.
    ///
    /// The questionnaire schema is validated up front; validation errors are
    /// mapped to a [`FrameworkError`](llm_core::FrameworkError).
    pub fn start_questionnaire(
        &self,
        questionnaire: Questionnaire,
    ) -> Result<QuestionnaireRun> {
        QuestionnaireRun::new(questionnaire).map_err(|errors| {
            llm_core::FrameworkError::questionnaire(errors.join("; "))
        })
    }

    /// Convert a completed set of questionnaire answers into a
    /// [`SessionConfig`] for the given provider.
    ///
    /// This performs a best-effort mapping from well-known question ids to
    /// config fields:
    ///
    /// - `"model"` (choice) -> `SessionConfig::model`
    /// - `"system_prompt"` (text) -> `SessionConfig::system_prompt`
    /// - `"max_turns"` (number) -> `SessionConfig::limits.max_turns`
    ///
    /// All other fields are left at their defaults.
    pub fn answers_to_session_config(
        &self,
        answers: &AnswerMap,
        provider_id: ProviderId,
    ) -> SessionConfig {
        let model = answers
            .choice(&QuestionId::new("model"))
            .map(llm_core::ModelId::new);

        let system_prompt = answers
            .text(&QuestionId::new("system_prompt"))
            .map(|s| s.to_owned());

        let max_turns = answers
            .number(&QuestionId::new("max_turns"))
            .map(|n| n as usize);

        let mut limits = llm_session::SessionLimits::default();
        if let Some(mt) = max_turns {
            limits.max_turns = mt;
        }

        SessionConfig {
            provider_id,
            model,
            system_prompt,
            tool_policy: ToolPolicy::default(),
            limits,
            metadata: Default::default(),
        }
    }
}

impl Default for QuestionnaireService {
    fn default() -> Self {
        Self::new()
    }
}
