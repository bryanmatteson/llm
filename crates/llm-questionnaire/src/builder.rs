//! Builder for constructing [`Questionnaire`] and [`Question`] values.
//!
//! # Example
//!
//! ```
//! use llm_questionnaire::QuestionnaireBuilder;
//!
//! let q = QuestionnaireBuilder::new("onboarding", "Onboarding")
//!     .description("Set up your account")
//!     .choice_with("role", "What is your role?", &["engineer", "designer", "pm"], |q| {
//!         q.default("engineer").required()
//!     })
//!     .yes_no_with("notifications", "Enable notifications?", |q| q.default_yes())
//!     .text_with("name", "What is your name?", |q| {
//!         q.placeholder("Jane Doe").help("Your display name").required().min_length(2)
//!     })
//!     .number_with("age", "How old are you?", |q| q.range(13.0, 120.0))
//!     .text("notes", "Anything else?")
//!     .build();
//! ```

use crate::condition::ConditionExpr;
use crate::ids::{QuestionId, QuestionnaireId};
use crate::schema::{ChoiceOption, Question, QuestionKind, Questionnaire};
use crate::validate::ValidationRule;

// ---------------------------------------------------------------------------
// QuestionnaireBuilder
// ---------------------------------------------------------------------------

/// Builder for a [`Questionnaire`].
///
/// Questions are added via type-specific methods. Each comes in two forms:
///
/// - **With closure** — `.choice("id", "label", opts, |q| q.default("a").required())`
/// - **Bare** — `.choice_bare("id", "label", opts)` when no configuration is needed
///
/// For question types that don't require extra positional args (yes/no, text,
/// number), the bare form drops the `_bare` suffix and the closure form uses
/// the same name — Rust disambiguates by arity.
pub struct QuestionnaireBuilder {
    id: QuestionnaireId,
    title: String,
    description: String,
    questions: Vec<Question>,
}

impl QuestionnaireBuilder {
    /// Start building a new questionnaire.
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: QuestionnaireId::new(id.into()),
            title: title.into(),
            description: String::new(),
            questions: Vec::new(),
        }
    }

    /// Set the questionnaire description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    // -- internal helper -----------------------------------------------------

    fn push(
        mut self,
        kind: QuestionKind,
        id: impl Into<String>,
        label: impl Into<String>,
        configure: impl FnOnce(QuestionConfig) -> QuestionConfig,
    ) -> Self {
        self.questions
            .push(configure(QuestionConfig::new(id, label, kind)).into_question());
        self
    }

    // -- choice --------------------------------------------------------------

    /// Add a choice question.
    pub fn choice(self, id: impl Into<String>, label: impl Into<String>, options: &[&str]) -> Self {
        self.push(Self::choice_kind(options), id, label, |q| q)
    }

    /// Add a choice question with a configuration closure.
    pub fn choice_with(
        self,
        id: impl Into<String>,
        label: impl Into<String>,
        options: &[&str],
        configure: impl FnOnce(QuestionConfig) -> QuestionConfig,
    ) -> Self {
        self.push(Self::choice_kind(options), id, label, configure)
    }

    /// Add a choice question with explicit `(value, label)` pairs and a configuration closure.
    pub fn choice_labeled(
        self,
        id: impl Into<String>,
        label: impl Into<String>,
        options: &[(&str, &str)],
        configure: impl FnOnce(QuestionConfig) -> QuestionConfig,
    ) -> Self {
        let opts = options
            .iter()
            .map(|(v, l)| ChoiceOption {
                value: (*v).into(),
                label: (*l).into(),
            })
            .collect();
        self.push(
            QuestionKind::Choice {
                options: opts,
                default: None,
            },
            id,
            label,
            configure,
        )
    }

    fn choice_kind(options: &[&str]) -> QuestionKind {
        let opts = options
            .iter()
            .map(|v| ChoiceOption {
                value: (*v).into(),
                label: (*v).into(),
            })
            .collect();
        QuestionKind::Choice {
            options: opts,
            default: None,
        }
    }

    // -- yes/no --------------------------------------------------------------

    /// Add a yes/no question.
    pub fn yes_no(self, id: impl Into<String>, label: impl Into<String>) -> Self {
        self.push(QuestionKind::YesNo { default: None }, id, label, |q| q)
    }

    /// Add a yes/no question with a configuration closure.
    pub fn yes_no_with(
        self,
        id: impl Into<String>,
        label: impl Into<String>,
        configure: impl FnOnce(QuestionConfig) -> QuestionConfig,
    ) -> Self {
        self.push(QuestionKind::YesNo { default: None }, id, label, configure)
    }

    // -- text ----------------------------------------------------------------

    /// Add a text question.
    pub fn text(self, id: impl Into<String>, label: impl Into<String>) -> Self {
        self.push(QuestionKind::Text { placeholder: None }, id, label, |q| q)
    }

    /// Add a text question with a configuration closure.
    pub fn text_with(
        self,
        id: impl Into<String>,
        label: impl Into<String>,
        configure: impl FnOnce(QuestionConfig) -> QuestionConfig,
    ) -> Self {
        self.push(
            QuestionKind::Text { placeholder: None },
            id,
            label,
            configure,
        )
    }

    // -- number --------------------------------------------------------------

    /// Add a number question.
    pub fn number(self, id: impl Into<String>, label: impl Into<String>) -> Self {
        self.push(
            QuestionKind::Number {
                min: None,
                max: None,
                default: None,
            },
            id,
            label,
            |q| q,
        )
    }

    /// Add a number question with a configuration closure.
    pub fn number_with(
        self,
        id: impl Into<String>,
        label: impl Into<String>,
        configure: impl FnOnce(QuestionConfig) -> QuestionConfig,
    ) -> Self {
        self.push(
            QuestionKind::Number {
                min: None,
                max: None,
                default: None,
            },
            id,
            label,
            configure,
        )
    }

    // -- multi-select --------------------------------------------------------

    /// Add a multi-select question.
    pub fn multi_select(
        self,
        id: impl Into<String>,
        label: impl Into<String>,
        options: &[&str],
    ) -> Self {
        let opts = options
            .iter()
            .map(|v| ChoiceOption {
                value: (*v).into(),
                label: (*v).into(),
            })
            .collect();
        self.push(
            QuestionKind::MultiSelect { options: opts },
            id,
            label,
            |q| q,
        )
    }

    /// Add a multi-select question with a configuration closure.
    pub fn multi_select_with(
        self,
        id: impl Into<String>,
        label: impl Into<String>,
        options: &[&str],
        configure: impl FnOnce(QuestionConfig) -> QuestionConfig,
    ) -> Self {
        let opts = options
            .iter()
            .map(|v| ChoiceOption {
                value: (*v).into(),
                label: (*v).into(),
            })
            .collect();
        self.push(
            QuestionKind::MultiSelect { options: opts },
            id,
            label,
            configure,
        )
    }

    // -- raw -----------------------------------------------------------------

    /// Push a fully-built [`Question`] directly.
    pub fn question(mut self, question: Question) -> Self {
        self.questions.push(question);
        self
    }

    /// Finish building and return the [`Questionnaire`].
    pub fn build(self) -> Questionnaire {
        Questionnaire {
            id: self.id,
            title: self.title,
            description: self.description,
            questions: self.questions,
        }
    }
}

// ---------------------------------------------------------------------------
// QuestionConfig
// ---------------------------------------------------------------------------

/// Configuration handle for a single question, received inside the closure
/// passed to [`QuestionnaireBuilder`] methods.
///
/// Every setter returns `self` by value so calls can be chained.
pub struct QuestionConfig {
    id: QuestionId,
    label: String,
    kind: QuestionKind,
    required: bool,
    help_text: Option<String>,
    validation: Vec<ValidationRule>,
    condition: Option<ConditionExpr>,
}

impl QuestionConfig {
    fn new(id: impl Into<String>, label: impl Into<String>, kind: QuestionKind) -> Self {
        Self {
            id: QuestionId::new(id.into()),
            label: label.into(),
            kind,
            required: false,
            help_text: None,
            validation: Vec::new(),
            condition: None,
        }
    }

    fn into_question(self) -> Question {
        Question {
            id: self.id,
            label: self.label,
            help_text: self.help_text,
            kind: self.kind,
            required: self.required,
            validation: self.validation,
            condition: self.condition,
        }
    }

    /// Mark this question as required.
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Set help text displayed alongside the question.
    pub fn help(mut self, text: impl Into<String>) -> Self {
        self.help_text = Some(text.into());
        self
    }

    /// Set the default value for a choice question.
    pub fn default(mut self, value: impl Into<String>) -> Self {
        if let QuestionKind::Choice {
            ref mut default, ..
        } = self.kind
        {
            *default = Some(value.into());
        }
        self
    }

    /// Set default to `true` for a yes/no question.
    pub fn default_yes(mut self) -> Self {
        if let QuestionKind::YesNo { ref mut default } = self.kind {
            *default = Some(true);
        }
        self
    }

    /// Set default to `false` for a yes/no question.
    pub fn default_no(mut self) -> Self {
        if let QuestionKind::YesNo { ref mut default } = self.kind {
            *default = Some(false);
        }
        self
    }

    /// Set a default number value.
    pub fn default_number(mut self, n: f64) -> Self {
        if let QuestionKind::Number {
            ref mut default, ..
        } = self.kind
        {
            *default = Some(n);
        }
        self
    }

    /// Set a placeholder for a text question.
    pub fn placeholder(mut self, text: impl Into<String>) -> Self {
        if let QuestionKind::Text {
            ref mut placeholder,
        } = self.kind
        {
            *placeholder = Some(text.into());
        }
        self
    }

    /// Set the numeric range for a number question.
    pub fn range(mut self, lo: f64, hi: f64) -> Self {
        if let QuestionKind::Number {
            ref mut min,
            ref mut max,
            ..
        } = self.kind
        {
            *min = Some(lo);
            *max = Some(hi);
        }
        self
    }

    /// Add a minimum character-length validation rule.
    pub fn min_length(mut self, n: usize) -> Self {
        self.validation.push(ValidationRule::MinLength(n));
        self
    }

    /// Add a maximum character-length validation rule.
    pub fn max_length(mut self, n: usize) -> Self {
        self.validation.push(ValidationRule::MaxLength(n));
        self
    }

    /// Add a regex pattern validation rule.
    pub fn pattern(mut self, regex: impl Into<String>) -> Self {
        self.validation.push(ValidationRule::Pattern(regex.into()));
        self
    }

    /// Only show this question when another question's answer equals a value.
    pub fn show_if_equals(
        mut self,
        question_id: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.condition = Some(ConditionExpr::Equals {
            question_id: QuestionId::new(question_id.into()),
            value: serde_json::Value::String(value.into()),
        });
        self
    }

    /// Only show this question when another question has been answered.
    pub fn show_if_answered(mut self, question_id: impl Into<String>) -> Self {
        self.condition = Some(ConditionExpr::Answered {
            question_id: QuestionId::new(question_id.into()),
        });
        self
    }

    /// Set an arbitrary condition expression.
    pub fn condition(mut self, cond: ConditionExpr) -> Self {
        self.condition = Some(cond);
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_questionnaire() {
        let q = QuestionnaireBuilder::new("test", "Test Questionnaire")
            .description("A test")
            .yes_no_with("agree", "Do you agree?", |q| q.default_yes().required())
            .build();

        assert_eq!(q.id.as_str(), "test");
        assert_eq!(q.title, "Test Questionnaire");
        assert_eq!(q.description, "A test");
        assert_eq!(q.questions.len(), 1);
        assert!(q.questions[0].required);
        assert_eq!(
            q.questions[0].kind,
            QuestionKind::YesNo {
                default: Some(true)
            }
        );
    }

    #[test]
    fn build_with_choices_and_conditions() {
        let q = QuestionnaireBuilder::new("setup", "Setup")
            .choice_with(
                "provider",
                "Which provider?",
                &["openai", "anthropic"],
                |q| q.default("openai").required(),
            )
            .text_with("api_key", "API Key:", |q| {
                q.placeholder("sk-...")
                    .min_length(8)
                    .show_if_equals("provider", "openai")
                    .required()
            })
            .build();

        assert_eq!(q.questions.len(), 2);

        let q0 = &q.questions[0];
        assert_eq!(q0.id.as_str(), "provider");
        assert!(q0.required);
        match &q0.kind {
            QuestionKind::Choice { options, default } => {
                assert_eq!(options.len(), 2);
                assert_eq!(default.as_deref(), Some("openai"));
            }
            _ => panic!("expected Choice"),
        }

        let q1 = &q.questions[1];
        assert_eq!(q1.id.as_str(), "api_key");
        assert!(q1.condition.is_some());
        assert_eq!(q1.validation.len(), 1);
    }

    #[test]
    fn build_number_with_range() {
        let q = QuestionnaireBuilder::new("n", "Numbers")
            .number_with("age", "Your age?", |q| {
                q.range(0.0, 150.0).default_number(25.0)
            })
            .build();

        match &q.questions[0].kind {
            QuestionKind::Number { min, max, default } => {
                assert_eq!(*min, Some(0.0));
                assert_eq!(*max, Some(150.0));
                assert_eq!(*default, Some(25.0));
            }
            _ => panic!("expected Number"),
        }
    }

    #[test]
    fn build_multi_select() {
        let q = QuestionnaireBuilder::new("ms", "Multi")
            .multi_select_with(
                "tools",
                "Which tools?",
                &["search", "calc", "weather"],
                |q| q.help("Pick all that apply"),
            )
            .build();

        let q0 = &q.questions[0];
        assert_eq!(q0.help_text.as_deref(), Some("Pick all that apply"));
        match &q0.kind {
            QuestionKind::MultiSelect { options } => assert_eq!(options.len(), 3),
            _ => panic!("expected MultiSelect"),
        }
    }

    #[test]
    fn builder_output_passes_schema_validation() {
        let q = QuestionnaireBuilder::new("valid", "Valid")
            .choice_with("c", "Pick one", &["a", "b"], |q| q.default("a"))
            .yes_no("yn", "Yes?")
            .build();

        let run = crate::engine::QuestionnaireRun::new(q);
        assert!(run.is_ok());
    }

    #[test]
    fn choice_with_labels() {
        let q = QuestionnaireBuilder::new("l", "Labels")
            .choice_labeled(
                "lang",
                "Language",
                &[("en", "English"), ("es", "Spanish")],
                |q| q,
            )
            .build();

        match &q.questions[0].kind {
            QuestionKind::Choice { options, .. } => {
                assert_eq!(options[0].value, "en");
                assert_eq!(options[0].label, "English");
            }
            _ => panic!("expected Choice"),
        }
    }

    #[test]
    fn bare_methods_add_unconfigured_questions() {
        let q = QuestionnaireBuilder::new("bare", "Bare")
            .text("name", "Name?")
            .yes_no("agree", "Agree?")
            .number("count", "How many?")
            .choice("color", "Color?", &["red", "blue"])
            .multi_select("tags", "Tags?", &["a", "b"])
            .build();

        assert_eq!(q.questions.len(), 5);
        for question in &q.questions {
            assert!(!question.required);
            assert!(question.help_text.is_none());
            assert!(question.condition.is_none());
            assert!(question.validation.is_empty());
        }
    }

    #[test]
    fn mixed_bare_and_configured() {
        let q = QuestionnaireBuilder::new("mix", "Mix")
            .text_with("name", "Name?", |q| q.required().min_length(2))
            .text("notes", "Notes?")
            .yes_no_with("agree", "Agree?", |q| q.default_yes())
            .build();

        assert!(q.questions[0].required);
        assert!(!q.questions[1].required);
        assert_eq!(
            q.questions[2].kind,
            QuestionKind::YesNo {
                default: Some(true)
            }
        );
    }
}
