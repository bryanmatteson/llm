use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ids::{QuestionId, QuestionnaireId, SectionId};

use crate::condition::ConditionExpr;
use crate::validate::ValidationRule;

/// A named group of questions within a questionnaire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Section {
    pub id: SectionId,
    pub title: String,
    #[serde(default)]
    pub description: String,
    pub questions: Vec<Question>,
}

/// A complete questionnaire definition.
///
/// Questions are organized into [`Section`]s. The flat [`questions`](Questionnaire::questions)
/// field contains all questions across all sections (in section order) for
/// backward compatibility and index-based engine access.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Questionnaire {
    pub id: QuestionnaireId,
    pub title: String,
    pub description: String,
    /// Sectioned grouping of questions. Each section has a title and its own
    /// questions. When empty, all questions live in the flat `questions` vec
    /// (backward-compatible mode).
    #[serde(default)]
    pub sections: Vec<Section>,
    /// Flat view of all questions (union of all sections' questions, in order).
    /// Maintained for backward compatibility — the engine indexes into this.
    pub questions: Vec<Question>,
}

/// A single question within a questionnaire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
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
        default: Option<String>,
    },
    Number {
        min: Option<f64>,
        max: Option<f64>,
        default: Option<f64>,
    },
    MultiSelect {
        options: Vec<ChoiceOption>,
        default: Option<Vec<String>>,
    },
    /// A read-only informational block displayed to the user.
    /// Not collected in the answer map — purely for display.
    Info {
        content: String,
    },
}

/// A selectable option within a Choice or MultiSelect question.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, JsonSchema)]
pub struct ChoiceOption {
    pub value: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl Questionnaire {
    /// Build a questionnaire from sections, populating the flat `questions`
    /// vec automatically.
    pub fn from_sections(
        id: QuestionnaireId,
        title: String,
        description: String,
        sections: Vec<Section>,
    ) -> Self {
        let questions = sections.iter().flat_map(|s| s.questions.clone()).collect();
        Self {
            id,
            title,
            description,
            sections,
            questions,
        }
    }

    /// Returns the section that contains the question at the given flat index,
    /// or `None` if sections are empty.
    pub fn section_of_index(&self, flat_index: usize) -> Option<&Section> {
        let mut offset = 0;
        for section in &self.sections {
            let end = offset + section.questions.len();
            if flat_index < end {
                return Some(section);
            }
            offset = end;
        }
        None
    }

    /// Deserialize from a JSON value and validate the schema.
    ///
    /// This is the primary entry point for constructing a questionnaire
    /// dynamically — e.g. from an LLM tool call's JSON arguments.
    ///
    /// Handles both old format (flat `questions` only) and new format
    /// (with `sections`). When sections are empty, questions are wrapped
    /// in an implicit default section.
    pub fn from_value(value: serde_json::Value) -> Result<Self, Vec<String>> {
        let mut q: Self = serde_json::from_value(value)
            .map_err(|e| vec![format!("invalid questionnaire JSON: {e}")])?;
        // If sections are empty but questions exist, wrap in a default section.
        if q.sections.is_empty() && !q.questions.is_empty() {
            q.sections = vec![Section {
                id: SectionId::new(""),
                title: String::new(),
                description: String::new(),
                questions: q.questions.clone(),
            }];
        }
        crate::validate::validate_questionnaire_schema(&q)?;
        Ok(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn questionnaire_serde_roundtrip() {
        let questions = vec![
            Question {
                id: QuestionId::new("lang"),
                label: "Language".into(),
                help_text: Some("Pick your language".into()),
                kind: QuestionKind::Choice {
                    options: vec![
                        ChoiceOption {
                            value: "en".into(),
                            label: "English".into(),
                            description: None,
                        },
                        ChoiceOption {
                            value: "es".into(),
                            label: "Spanish".into(),
                            description: None,
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
                kind: QuestionKind::YesNo {
                    default: Some(true),
                },
                required: true,
                validation: vec![],
                condition: None,
            },
        ];
        let q = Questionnaire {
            id: QuestionnaireId::new("setup"),
            title: "Setup".into(),
            description: "Initial setup".into(),
            sections: vec![],
            questions,
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
                default: None,
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
                    description: None,
                }],
                default: None,
            },
            QuestionKind::Info {
                content: "preview text".into(),
            },
        ];

        for kind in &kinds {
            let json = serde_json::to_string(kind).unwrap();
            let _back: QuestionKind = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn question_kind_serde_uses_snake_case() {
        let kind = QuestionKind::YesNo { default: None };
        let json = serde_json::to_string(&kind).unwrap();
        assert!(json.contains(r#""type":"yes_no""#));

        let kind = QuestionKind::MultiSelect {
            options: vec![],
            default: None,
        };
        let json = serde_json::to_string(&kind).unwrap();
        assert!(json.contains(r#""type":"multi_select""#));
    }

    #[test]
    fn from_value_valid() {
        let json = serde_json::json!({
            "id": "test",
            "title": "Test",
            "description": "desc",
            "questions": [{
                "id": "q1",
                "label": "Pick one",
                "kind": {"type": "choice", "options": [{"value": "a", "label": "A"}], "default": null},
                "required": false,
                "validation": [],
                "condition": null
            }]
        });
        let q = Questionnaire::from_value(json).unwrap();
        assert_eq!(q.questions.len(), 1);
    }

    #[test]
    fn from_value_invalid_json() {
        let json = serde_json::json!({"not": "a questionnaire"});
        let errs = Questionnaire::from_value(json).unwrap_err();
        assert!(errs[0].contains("invalid questionnaire JSON"));
    }

    #[test]
    fn from_value_schema_invalid() {
        let json = serde_json::json!({
            "id": "bad",
            "title": "Bad",
            "description": "desc",
            "questions": [
                {"id": "dup", "label": "Q1", "kind": {"type": "text", "placeholder": null, "default": null}, "required": false, "validation": [], "condition": null},
                {"id": "dup", "label": "Q2", "kind": {"type": "text", "placeholder": null, "default": null}, "required": false, "validation": [], "condition": null}
            ]
        });
        let errs = Questionnaire::from_value(json).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("duplicate")));
    }

    #[test]
    fn choice_option_description_preserved() {
        let opt = ChoiceOption {
            value: "a".into(),
            label: "A".into(),
            description: Some("The first option".into()),
        };
        let json = serde_json::to_string(&opt).unwrap();
        assert!(json.contains("The first option"));
        let back: ChoiceOption = serde_json::from_str(&json).unwrap();
        assert_eq!(back.description.as_deref(), Some("The first option"));
    }

    #[test]
    fn choice_option_description_absent() {
        // description is optional; absent in JSON should parse as None
        let json = r#"{"value":"a","label":"A"}"#;
        let opt: ChoiceOption = serde_json::from_str(json).unwrap();
        assert_eq!(opt.description, None);
    }

    #[test]
    fn text_default_roundtrip() {
        let kind = QuestionKind::Text {
            placeholder: None,
            default: Some("hello".into()),
        };
        let json = serde_json::to_string(&kind).unwrap();
        let back: QuestionKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, kind);
    }

    #[test]
    fn multi_select_default_roundtrip() {
        let kind = QuestionKind::MultiSelect {
            options: vec![
                ChoiceOption {
                    value: "a".into(),
                    label: "A".into(),
                    description: None,
                },
                ChoiceOption {
                    value: "b".into(),
                    label: "B".into(),
                    description: None,
                },
            ],
            default: Some(vec!["a".into()]),
        };
        let json = serde_json::to_string(&kind).unwrap();
        let back: QuestionKind = serde_json::from_str(&json).unwrap();
        assert_eq!(back, kind);
    }
}
