use serde::{Deserialize, Serialize};

use llm_core::{QuestionId, QuestionnaireId};

use crate::condition::ConditionExpr;
use crate::validate::ValidationRule;

/// A complete questionnaire definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Questionnaire {
    pub id: QuestionnaireId,
    pub title: String,
    pub description: String,
    pub questions: Vec<Question>,
}

/// A single question within a questionnaire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Question {
    pub id: QuestionId,
    pub label: String,
    pub help_text: Option<String>,
    pub kind: QuestionKind,
    pub required: bool,
    pub validation: Vec<ValidationRule>,
    /// When present, the question is only shown if this condition evaluates to true.
    pub condition: Option<ConditionExpr>,
}

/// The kind of input expected for a question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum QuestionKind {
    Choice {
        options: Vec<ChoiceOption>,
        default: Option<String>,
    },
    YesNo {
        default: Option<bool>,
    },
    Text {
        placeholder: Option<String>,
    },
    Number {
        min: Option<f64>,
        max: Option<f64>,
        default: Option<f64>,
    },
    MultiSelect {
        options: Vec<ChoiceOption>,
    },
}

/// A selectable option within a Choice or MultiSelect question.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChoiceOption {
    pub value: String,
    pub label: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn questionnaire_serde_roundtrip() {
        let q = Questionnaire {
            id: QuestionnaireId::new("setup"),
            title: "Setup".into(),
            description: "Initial setup".into(),
            questions: vec![
                Question {
                    id: QuestionId::new("lang"),
                    label: "Language".into(),
                    help_text: Some("Pick your language".into()),
                    kind: QuestionKind::Choice {
                        options: vec![
                            ChoiceOption {
                                value: "en".into(),
                                label: "English".into(),
                            },
                            ChoiceOption {
                                value: "es".into(),
                                label: "Spanish".into(),
                            },
                        ],
                        default: Some("en".into()),
                    },
                    required: true,
                    validation: vec![],
                    condition: None,
                },
                Question {
                    id: QuestionId::new("agree"),
                    label: "Do you agree?".into(),
                    help_text: None,
                    kind: QuestionKind::YesNo { default: Some(true) },
                    required: true,
                    validation: vec![],
                    condition: None,
                },
            ],
        };

        let json = serde_json::to_string_pretty(&q).unwrap();
        let back: Questionnaire = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id.as_str(), "setup");
        assert_eq!(back.questions.len(), 2);
    }

    #[test]
    fn question_kind_variants() {
        let kinds = vec![
            QuestionKind::Choice {
                options: vec![],
                default: None,
            },
            QuestionKind::YesNo { default: None },
            QuestionKind::Text {
                placeholder: Some("enter text".into()),
            },
            QuestionKind::Number {
                min: Some(0.0),
                max: Some(100.0),
                default: Some(50.0),
            },
            QuestionKind::MultiSelect {
                options: vec![ChoiceOption {
                    value: "a".into(),
                    label: "A".into(),
                }],
            },
        ];

        for kind in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            let _back: QuestionKind = serde_json::from_str(&json).unwrap();
        }
    }
}
