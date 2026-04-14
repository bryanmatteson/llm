use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use llm_core::Result;
use llm_tools::ToolContext;
use llm_tools::tool::{Tool, ToolInfo};

use crate::answer::AnswerMap;
use crate::condition::ConditionExpr;
use crate::ids::{QuestionId, QuestionnaireId};
use crate::schema::{ChoiceOption, Question, QuestionKind, Questionnaire};

// ---------------------------------------------------------------------------
// Tool‑facing input types
// ---------------------------------------------------------------------------

/// Input schema for the `ask_questions` tool.
///
/// The LLM constructs this to present a batch of questions to the user.
/// Questions may include conditions for progressive disclosure — if a prior
/// answer makes a question irrelevant, the engine skips it automatically.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AskQuestionsInput {
    /// The questions to present. Batch related questions together;
    /// maximum 8 per call. May call the tool multiple times for follow‑ups.
    pub questions: Vec<QuestionInput>,
}

/// A single question within an `ask_questions` tool call.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QuestionInput {
    /// Unique identifier for this question. Used as the key in the answers map.
    pub id: String,
    /// The question text shown to the user.
    pub prompt: String,
    /// The type of answer expected.
    #[serde(rename = "type")]
    pub kind: QuestionType,
    /// Available options for `choice` and `multi_choice` questions.
    /// Each option has a machine‑readable `value`, a human `label`,
    /// and an optional `description` for context.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<OptionInput>,
    /// Default value if the user accepts without changing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    /// Whether this question must be answered. Defaults to false.
    #[serde(default)]
    pub required: bool,
    /// Minimum value for `number` questions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Maximum value for `number` questions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Condition for progressive disclosure: skip this question when a
    /// prior answer makes it irrelevant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<SkipCondition>,
}

/// The type of answer expected from the user.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QuestionType {
    /// Pick exactly one option.
    Choice,
    /// Pick one or more options.
    MultiChoice,
    /// Yes or no.
    YesNo,
    /// Free‑form text.
    Text,
    /// Numeric value, optionally bounded by min/max.
    Number,
}

/// A selectable option for `choice` / `multi_choice` questions.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OptionInput {
    /// Machine‑readable value returned in the answer.
    pub value: String,
    /// Human‑readable label shown to the user.
    pub label: String,
    /// Optional description providing context for this option.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Skip this question when a prior answer makes it irrelevant.
///
/// The question is **shown** only when the answer to `question_id`
/// equals `equals`. Otherwise it is skipped.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SkipCondition {
    /// The ID of a prior question to check.
    pub question_id: String,
    /// The expected value. The question is shown only if the prior
    /// answer matches this value.
    pub equals: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Tool output
// ---------------------------------------------------------------------------

/// Output from the `ask_questions` tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AskQuestionsOutput {
    /// Map of question ID → answer value.
    pub answers: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Conversions: tool input → Questionnaire, AnswerMap → tool output
// ---------------------------------------------------------------------------

/// Maximum number of questions allowed per tool call.
pub const MAX_QUESTIONS_PER_CALL: usize = 8;

impl AskQuestionsInput {
    /// Convert into the internal [`Questionnaire`] representation,
    /// validating the schema in the process.
    pub fn into_questionnaire(self) -> std::result::Result<Questionnaire, Vec<String>> {
        if self.questions.len() > MAX_QUESTIONS_PER_CALL {
            return Err(vec![format!(
                "too many questions ({}) — maximum {} per call",
                self.questions.len(),
                MAX_QUESTIONS_PER_CALL
            )]);
        }

        let questions: Vec<Question> = self
            .questions
            .into_iter()
            .map(|q| q.into_question())
            .collect();

        let questionnaire = Questionnaire {
            id: QuestionnaireId::new("ask_questions"),
            title: String::new(),
            description: String::new(),
            sections: vec![],
            questions,
        };

        crate::validate::validate_questionnaire_schema(&questionnaire)?;
        Ok(questionnaire)
    }
}

impl QuestionInput {
    fn into_question(self) -> Question {
        let kind = match self.kind {
            QuestionType::Choice => QuestionKind::Choice {
                options: self.options.iter().map(|o| o.to_choice_option()).collect(),
                default: self
                    .default
                    .as_ref()
                    .and_then(|v| v.as_str())
                    .map(Into::into),
            },
            QuestionType::MultiChoice => QuestionKind::MultiSelect {
                options: self.options.iter().map(|o| o.to_choice_option()).collect(),
                default: self.default.as_ref().and_then(|v| {
                    v.as_array().map(|arr| {
                        arr.iter()
                            .filter_map(|item| item.as_str().map(String::from))
                            .collect()
                    })
                }),
            },
            QuestionType::YesNo => QuestionKind::YesNo {
                default: self.default.as_ref().and_then(|v| v.as_bool()),
            },
            QuestionType::Text => QuestionKind::Text {
                placeholder: None,
                default: self
                    .default
                    .as_ref()
                    .and_then(|v| v.as_str())
                    .map(Into::into),
            },
            QuestionType::Number => QuestionKind::Number {
                min: self.min,
                max: self.max,
                default: self.default.as_ref().and_then(|v| v.as_f64()),
            },
        };

        let condition = self.condition.map(|c| ConditionExpr::Equals {
            question_id: QuestionId::new(c.question_id),
            value: c.equals,
        });

        Question {
            id: QuestionId::new(self.id),
            label: self.prompt,
            help_text: None,
            kind,
            required: self.required,
            validation: vec![],
            condition,
        }
    }
}

impl OptionInput {
    fn to_choice_option(&self) -> ChoiceOption {
        ChoiceOption {
            value: self.value.clone(),
            label: self.label.clone(),
            description: self.description.clone(),
        }
    }
}

impl AskQuestionsOutput {
    /// Build the output from a completed [`AnswerMap`].
    pub fn from_answer_map(answers: &AnswerMap) -> Self {
        let map: HashMap<String, serde_json::Value> = answers
            .iter()
            .map(|(id, val)| (id.as_str().to_owned(), val.to_json_value()))
            .collect();
        Self { answers: map }
    }
}

// ---------------------------------------------------------------------------
// QuestionHandler — pluggable user interaction
// ---------------------------------------------------------------------------

/// Trait implemented by the host application (CLI, GUI, etc.) to present
/// questions to the user and collect answers.
///
/// The handler receives a validated [`Questionnaire`] and must return a
/// complete [`AnswerMap`] with answers for all visible questions.
#[async_trait]
pub trait QuestionHandler: Send + Sync {
    async fn ask(&self, questionnaire: &Questionnaire) -> Result<AnswerMap>;
}

// ---------------------------------------------------------------------------
// AskQuestionsTool — the Tool implementation
// ---------------------------------------------------------------------------

const TOOL_DESCRIPTION: &str = "\
Ask the user a batch of questions and collect their answers.

Behavioral rules:
- Batch related questions together; do not scatter them across multiple calls.
- Do not ask for information that is already known or inferable from context.
- Maximum 8 questions per call. Call the tool again for follow‑ups.
- Dynamically populate options from workspace state when applicable.
- Use conditions for progressive disclosure — skip questions when a prior \
  answer makes them irrelevant.";

/// A first‑class tool that lets the LLM ask the user structured questions.
///
/// The tool converts the LLM's JSON input into a [`Questionnaire`], delegates
/// to a [`QuestionHandler`] for user interaction, and returns the answers.
pub struct AskQuestionsTool {
    handler: Arc<dyn QuestionHandler>,
}

impl fmt::Debug for AskQuestionsTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AskQuestionsTool").finish()
    }
}

impl AskQuestionsTool {
    pub fn new(handler: Arc<dyn QuestionHandler>) -> Self {
        Self { handler }
    }
}

#[async_trait]
impl Tool for AskQuestionsTool {
    type Input = AskQuestionsInput;
    type Output = AskQuestionsOutput;

    fn info(&self) -> ToolInfo {
        ToolInfo::new("ask_questions", TOOL_DESCRIPTION).display_name("Ask Questions")
    }

    async fn execute(
        &self,
        input: AskQuestionsInput,
        _ctx: &ToolContext,
    ) -> Result<AskQuestionsOutput> {
        let questionnaire = input
            .into_questionnaire()
            .map_err(|errs| llm_core::FrameworkError::questionnaire(errs.join("; ")))?;

        let answers = self.handler.ask(&questionnaire).await?;
        Ok(AskQuestionsOutput::from_answer_map(&answers))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answer::AnswerValue;
    use serde_json::json;

    fn qi(id: &str, prompt: &str, kind: QuestionType) -> QuestionInput {
        QuestionInput {
            id: id.into(),
            prompt: prompt.into(),
            kind,
            options: vec![],
            default: None,
            required: false,
            min: None,
            max: None,
            condition: None,
        }
    }

    #[test]
    fn input_serde_roundtrip() {
        let mut lang = qi("lang", "What language?", QuestionType::Choice);
        lang.options = vec![
            OptionInput {
                value: "rust".into(),
                label: "Rust".into(),
                description: Some("Systems programming".into()),
            },
            OptionInput {
                value: "python".into(),
                label: "Python".into(),
                description: None,
            },
        ];
        lang.default = Some(json!("rust"));

        let mut confirm = qi("confirm", "Proceed?", QuestionType::YesNo);
        confirm.default = Some(json!(true));

        let input = AskQuestionsInput {
            questions: vec![lang, confirm],
        };
        let json_str = serde_json::to_string_pretty(&input).unwrap();
        let back: AskQuestionsInput = serde_json::from_str(&json_str).unwrap();
        assert_eq!(back.questions.len(), 2);
        assert_eq!(back.questions[0].id, "lang");
        assert_eq!(back.questions[1].kind, QuestionType::YesNo);
    }

    #[test]
    fn into_questionnaire_valid() {
        let mut q = qi("color", "Favorite color?", QuestionType::Choice);
        q.options = vec![
            OptionInput {
                value: "red".into(),
                label: "Red".into(),
                description: None,
            },
            OptionInput {
                value: "blue".into(),
                label: "Blue".into(),
                description: None,
            },
        ];
        q.default = Some(json!("red"));
        let input = AskQuestionsInput { questions: vec![q] };

        let questionnaire = input.into_questionnaire().unwrap();
        assert_eq!(questionnaire.questions.len(), 1);
        assert_eq!(questionnaire.questions[0].id.as_str(), "color");
    }

    #[test]
    fn into_questionnaire_rejects_duplicate_ids() {
        let input = AskQuestionsInput {
            questions: vec![
                qi("dup", "Q1", QuestionType::Text),
                qi("dup", "Q2", QuestionType::Text),
            ],
        };
        let err = input.into_questionnaire().unwrap_err();
        assert!(err.iter().any(|e| e.contains("duplicate")));
    }

    #[test]
    fn condition_converts_to_equals() {
        let mut ci = qi("ci_provider", "Which CI provider?", QuestionType::Choice);
        ci.options = vec![OptionInput {
            value: "github".into(),
            label: "GitHub Actions".into(),
            description: None,
        }];
        ci.condition = Some(SkipCondition {
            question_id: "use_ci".into(),
            equals: json!(true),
        });

        let input = AskQuestionsInput {
            questions: vec![qi("use_ci", "Enable CI?", QuestionType::YesNo), ci],
        };
        let q = input.into_questionnaire().unwrap();
        assert!(q.questions[1].condition.is_some());
    }

    #[test]
    fn output_from_answer_map() {
        let mut answers = AnswerMap::new();
        answers.insert(QuestionId::new("lang"), AnswerValue::Choice("rust".into()));
        answers.insert(QuestionId::new("confirm"), AnswerValue::YesNo(true));

        let output = AskQuestionsOutput::from_answer_map(&answers);
        assert_eq!(output.answers["lang"], json!("rust"));
        assert_eq!(output.answers["confirm"], json!(true));
    }

    #[test]
    fn json_schema_generates() {
        let schema = schemars::schema_for!(AskQuestionsInput);
        let json = serde_json::to_value(&schema).unwrap();
        let schema_str = serde_json::to_string_pretty(&json).unwrap();
        assert!(schema_str.contains("questions"));
        assert!(schema_str.contains("QuestionType"));
    }

    #[test]
    fn text_default_preserved() {
        let mut q = qi("dir", "Output directory?", QuestionType::Text);
        q.default = Some(json!("specs"));
        let input = AskQuestionsInput { questions: vec![q] };
        let questionnaire = input.into_questionnaire().unwrap();
        match &questionnaire.questions[0].kind {
            QuestionKind::Text { default, .. } => assert_eq!(default.as_deref(), Some("specs")),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn multi_choice_default_preserved() {
        let mut q = qi("tags", "Select tags", QuestionType::MultiChoice);
        q.options = vec![
            OptionInput {
                value: "a".into(),
                label: "A".into(),
                description: None,
            },
            OptionInput {
                value: "b".into(),
                label: "B".into(),
                description: None,
            },
        ];
        q.default = Some(json!(["a"]));
        let input = AskQuestionsInput { questions: vec![q] };
        let questionnaire = input.into_questionnaire().unwrap();
        match &questionnaire.questions[0].kind {
            QuestionKind::MultiSelect { default, .. } => {
                assert_eq!(default.as_ref().unwrap(), &vec!["a".to_string()]);
            }
            _ => panic!("expected MultiSelect"),
        }
    }

    #[test]
    fn yes_no_default_false() {
        let mut q = qi("opt_out", "Opt out?", QuestionType::YesNo);
        q.default = Some(json!(false));
        let input = AskQuestionsInput { questions: vec![q] };
        let questionnaire = input.into_questionnaire().unwrap();
        match &questionnaire.questions[0].kind {
            QuestionKind::YesNo { default } => assert_eq!(*default, Some(false)),
            _ => panic!("expected YesNo"),
        }
    }

    #[test]
    fn description_threaded_through_conversion() {
        let mut q = qi("lang", "Language?", QuestionType::Choice);
        q.options = vec![OptionInput {
            value: "rs".into(),
            label: "Rust".into(),
            description: Some("Systems lang".into()),
        }];
        let input = AskQuestionsInput { questions: vec![q] };
        let questionnaire = input.into_questionnaire().unwrap();
        match &questionnaire.questions[0].kind {
            QuestionKind::Choice { options, .. } => {
                assert_eq!(options[0].description.as_deref(), Some("Systems lang"));
            }
            _ => panic!("expected Choice"),
        }
    }

    #[test]
    fn rejects_more_than_8_questions() {
        let questions: Vec<QuestionInput> = (0..9)
            .map(|i| qi(&format!("q{i}"), &format!("Q{i}?"), QuestionType::Text))
            .collect();
        let input = AskQuestionsInput { questions };
        let err = input.into_questionnaire().unwrap_err();
        assert!(err[0].contains("too many questions"));
    }

    #[test]
    fn accepts_exactly_8_questions() {
        let questions: Vec<QuestionInput> = (0..8)
            .map(|i| qi(&format!("q{i}"), &format!("Q{i}?"), QuestionType::Text))
            .collect();
        let input = AskQuestionsInput { questions };
        assert!(input.into_questionnaire().is_ok());
    }

    #[test]
    fn number_question_conversion() {
        let mut q = qi("age", "How old?", QuestionType::Number);
        q.min = Some(0.0);
        q.max = Some(150.0);
        q.default = Some(json!(25.0));
        let input = AskQuestionsInput { questions: vec![q] };
        let questionnaire = input.into_questionnaire().unwrap();
        match &questionnaire.questions[0].kind {
            QuestionKind::Number { min, max, default } => {
                assert_eq!(*min, Some(0.0));
                assert_eq!(*max, Some(150.0));
                assert_eq!(*default, Some(25.0));
            }
            _ => panic!("expected Number"),
        }
    }

    #[test]
    fn required_field_preserved() {
        let mut q = qi("name", "Your name?", QuestionType::Text);
        q.required = true;
        let input = AskQuestionsInput { questions: vec![q] };
        let questionnaire = input.into_questionnaire().unwrap();
        assert!(questionnaire.questions[0].required);
    }

    #[test]
    fn realistic_workspace_driven_call() {
        // Simulates an LLM call where options are populated from workspace state:
        // - Detected specs/ dir → default "specs" for spec_dir
        // - 4 discovered crates → governed_areas options
        // - .github/workflows/ present → CI enforcement option
        // - No LSP detected → L1 default for governance level
        let call_json = json!({
            "questions": [
                {
                    "id": "governance_level",
                    "prompt": "What level of governance do you need?",
                    "type": "choice",
                    "options": [
                        {"value": "L1", "label": "L1 — Lint only", "description": "Cargo clippy + fmt checks"},
                        {"value": "L2", "label": "L2 — Lint + specs", "description": "Plus specification enforcement"},
                        {"value": "L3", "label": "L3 — Full", "description": "Lint, specs, and runtime policy"}
                    ],
                    "default": "L1"
                },
                {
                    "id": "governed_areas",
                    "prompt": "Which crates should be governed?",
                    "type": "multi_choice",
                    "options": [
                        {"value": "llm-core", "label": "llm-core"},
                        {"value": "llm-tools", "label": "llm-tools"},
                        {"value": "llm-session", "label": "llm-session"},
                        {"value": "llm-mcp", "label": "llm-mcp"}
                    ],
                    "condition": {"question_id": "governance_level", "equals": "L2"}
                },
                {
                    "id": "spec_dir",
                    "prompt": "Where should specs live?",
                    "type": "text",
                    "default": "specs",
                    "condition": {"question_id": "governance_level", "equals": "L2"}
                },
                {
                    "id": "ci_enforce",
                    "prompt": "Enforce governance in CI?",
                    "type": "yes_no",
                    "default": true
                }
            ]
        });

        let input: AskQuestionsInput = serde_json::from_value(call_json).unwrap();
        assert_eq!(input.questions.len(), 4);

        let q = input.into_questionnaire().unwrap();
        assert_eq!(q.questions.len(), 4);
        // governed_areas and spec_dir have conditions
        assert!(q.questions[1].condition.is_some());
        assert!(q.questions[2].condition.is_some());
        // ci_enforce has no condition
        assert!(q.questions[3].condition.is_none());

        // Verify text default preserved (spec_dir = "specs")
        match &q.questions[2].kind {
            QuestionKind::Text { default, .. } => {
                assert_eq!(default.as_deref(), Some("specs"));
            }
            _ => panic!("expected Text for spec_dir"),
        }

        // Verify descriptions threaded through (governance_level options)
        match &q.questions[0].kind {
            QuestionKind::Choice { options, .. } => {
                assert_eq!(
                    options[0].description.as_deref(),
                    Some("Cargo clippy + fmt checks")
                );
            }
            _ => panic!("expected Choice for governance_level"),
        }

        // Verify yes_no default preserved
        match &q.questions[3].kind {
            QuestionKind::YesNo { default } => {
                assert_eq!(*default, Some(true));
            }
            _ => panic!("expected YesNo for ci_enforce"),
        }
    }
}
