use std::collections::HashSet;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::answer::AnswerValue;
use crate::schema::{ChoiceOption, QuestionKind, Questionnaire};

/// A rule applied to validate an individual answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum ValidationRule {
    /// The answer must be provided (non-empty text, non-null).
    Required,
    /// Minimum character length for text answers.
    MinLength(usize),
    /// Maximum character length for text answers.
    MaxLength(usize),
    /// A regex pattern that text answers must match.
    Pattern(String),
    /// Numeric answers must fall within the given range (inclusive).
    Range { min: f64, max: f64 },
}

/// Validate a single answer against the question kind and a set of validation rules.
///
/// Returns a (possibly empty) vec of human-readable error messages.
pub fn validate_answer(
    kind: &QuestionKind,
    rules: &[ValidationRule],
    answer: &AnswerValue,
) -> Vec<String> {
    let mut errors = Vec::new();

    // Check that the answer variant matches the question kind.
    match (kind, answer) {
        (QuestionKind::Choice { options, .. }, AnswerValue::Choice(v)) => {
            if !options.iter().any(|o| o.value == *v) {
                errors.push(format!(
                    "'{v}' is not a valid choice (expected one of: {})",
                    option_values_display(options)
                ));
            }
        }
        (QuestionKind::YesNo { .. }, AnswerValue::YesNo(_)) => {}
        (QuestionKind::Text { .. }, AnswerValue::Text(_)) => {}
        (QuestionKind::Number { min, max, .. }, AnswerValue::Number(n)) => {
            if let Some(lo) = min {
                if *n < *lo {
                    errors.push(format!("value {n} is below minimum {lo}"));
                }
            }
            if let Some(hi) = max {
                if *n > *hi {
                    errors.push(format!("value {n} is above maximum {hi}"));
                }
            }
        }
        (QuestionKind::MultiSelect { options, .. }, AnswerValue::MultiSelect(selected)) => {
            for s in selected {
                if !options.iter().any(|o| o.value == *s) {
                    errors.push(format!(
                        "'{s}' is not a valid selection (expected one of: {})",
                        option_values_display(options)
                    ));
                }
            }
        }
        _ => {
            errors.push(format!(
                "answer type does not match question kind: expected {:?}",
                kind_name(kind)
            ));
        }
    }

    // Apply explicit validation rules.
    for rule in rules {
        match rule {
            ValidationRule::Required => {
                let empty = match answer {
                    AnswerValue::Text(None) => true,
                    AnswerValue::Text(Some(s)) => s.trim().is_empty(),
                    AnswerValue::MultiSelect(v) => v.is_empty(),
                    _ => false,
                };
                if empty {
                    errors.push("this field is required".to_string());
                }
            }
            ValidationRule::MinLength(min) => {
                if let AnswerValue::Text(Some(s)) = answer {
                    if s.chars().count() < *min {
                        errors.push(format!(
                            "text is too short (minimum {min} characters, got {})",
                            s.chars().count()
                        ));
                    }
                }
            }
            ValidationRule::MaxLength(max) => {
                if let AnswerValue::Text(Some(s)) = answer {
                    if s.chars().count() > *max {
                        errors.push(format!(
                            "text is too long (maximum {max} characters, got {})",
                            s.chars().count()
                        ));
                    }
                }
            }
            ValidationRule::Pattern(pattern) => {
                if let AnswerValue::Text(Some(s)) = answer {
                    // Best-effort regex check; invalid patterns produce an error.
                    match regex_lite::Regex::new(pattern) {
                        Ok(re) => {
                            if !re.is_match(s) {
                                errors.push(format!(
                                    "text does not match required pattern '{pattern}'"
                                ));
                            }
                        }
                        Err(e) => {
                            errors.push(format!("invalid validation pattern '{pattern}': {e}"));
                        }
                    }
                }
            }
            ValidationRule::Range { min, max } => {
                if let AnswerValue::Number(n) = answer {
                    if *n < *min || *n > *max {
                        errors.push(format!("value {n} is outside range [{min}, {max}]"));
                    }
                }
            }
        }
    }

    errors
}

/// Recursively collect all question IDs referenced by a condition expression.
fn collect_condition_refs(cond: &crate::condition::ConditionExpr) -> Vec<&crate::ids::QuestionId> {
    use crate::condition::ConditionExpr;
    match cond {
        ConditionExpr::Equals { question_id, .. }
        | ConditionExpr::NotEquals { question_id, .. }
        | ConditionExpr::Answered { question_id } => vec![question_id],
        ConditionExpr::And(exprs) | ConditionExpr::Or(exprs) => {
            exprs.iter().flat_map(collect_condition_refs).collect()
        }
        ConditionExpr::Not(inner) => collect_condition_refs(inner),
    }
}

/// Validate the overall structure of a questionnaire definition.
///
/// Checks:
/// - No duplicate question IDs
/// - No empty question IDs or labels
/// - Choice defaults (if present) exist in the options list
/// - Choice options have unique values
/// - Choice / MultiSelect questions have at least one option
/// - Condition expressions reference existing question IDs
pub fn validate_questionnaire_schema(questionnaire: &Questionnaire) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let mut seen_ids = HashSet::new();

    // Collect all question IDs first for condition reference validation.
    let all_ids: HashSet<&str> = questionnaire
        .questions
        .iter()
        .map(|q| q.id.as_str())
        .collect();

    for question in &questionnaire.questions {
        let qid = question.id.as_str();

        // Duplicate ID check.
        if !seen_ids.insert(qid.to_string()) {
            errors.push(format!("duplicate question id '{qid}'"));
        }

        // Empty ID check.
        if qid.trim().is_empty() {
            errors.push("question id must not be empty".to_string());
        }

        // Empty label check.
        if question.label.trim().is_empty() {
            errors.push(format!("question '{qid}' has an empty label"));
        }

        // Condition reference check.
        if let Some(cond) = &question.condition {
            for ref_id in collect_condition_refs(cond) {
                if !all_ids.contains(ref_id.as_str()) {
                    errors.push(format!(
                        "question '{qid}' has a condition referencing unknown question '{ref_id}'"
                    ));
                }
            }
        }

        match &question.kind {
            QuestionKind::Choice { options, default } => {
                if options.is_empty() {
                    errors.push(format!(
                        "choice question '{qid}' must have at least one option"
                    ));
                }
                // Default must be in options.
                if let Some(def) = default {
                    if !options.iter().any(|o| o.value == *def) {
                        errors.push(format!(
                            "choice question '{qid}' has default '{def}' which is not in its options"
                        ));
                    }
                }
                // Unique option values.
                let mut seen_values = HashSet::new();
                for opt in options {
                    if !seen_values.insert(&opt.value) {
                        errors.push(format!(
                            "choice question '{qid}' has duplicate option value '{}'",
                            opt.value
                        ));
                    }
                }
            }
            QuestionKind::MultiSelect { options, default } => {
                if options.is_empty() {
                    errors.push(format!(
                        "multi-select question '{qid}' must have at least one option"
                    ));
                }
                let mut seen_values = HashSet::new();
                for opt in options {
                    if !seen_values.insert(&opt.value) {
                        errors.push(format!(
                            "multi-select question '{qid}' has duplicate option value '{}'",
                            opt.value
                        ));
                    }
                }
                // Defaults must all be in options.
                if let Some(defaults) = default {
                    for def in defaults {
                        if !options.iter().any(|o| o.value == *def) {
                            errors.push(format!(
                                "multi-select question '{qid}' has default '{def}' which is not in its options"
                            ));
                        }
                    }
                }
            }
            QuestionKind::Number { min, max, default } => {
                if let (Some(lo), Some(hi)) = (min, max) {
                    if lo > hi {
                        errors.push(format!(
                            "number question '{qid}' has min ({lo}) greater than max ({hi})"
                        ));
                    }
                }
                if let Some(def) = default {
                    if let Some(lo) = min {
                        if def < lo {
                            errors.push(format!(
                                "number question '{qid}' has default ({def}) below min ({lo})"
                            ));
                        }
                    }
                    if let Some(hi) = max {
                        if def > hi {
                            errors.push(format!(
                                "number question '{qid}' has default ({def}) above max ({hi})"
                            ));
                        }
                    }
                }
            }
            QuestionKind::YesNo { .. } | QuestionKind::Text { .. } => {}
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn option_values_display(options: &[ChoiceOption]) -> String {
    options
        .iter()
        .map(|o| o.value.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn kind_name(kind: &QuestionKind) -> &'static str {
    match kind {
        QuestionKind::Choice { .. } => "Choice",
        QuestionKind::YesNo { .. } => "YesNo",
        QuestionKind::Text { .. } => "Text",
        QuestionKind::Number { .. } => "Number",
        QuestionKind::MultiSelect { .. } => "MultiSelect",
    }
}

#[cfg(test)]
mod tests {
    use crate::ids::{QuestionId, QuestionnaireId};

    use super::*;
    use crate::schema::{ChoiceOption, Question, QuestionKind, Questionnaire};

    // -------- validate_answer tests --------

    fn opt(value: &str, label: &str) -> ChoiceOption {
        ChoiceOption {
            value: value.into(),
            label: label.into(),
            description: None,
        }
    }

    #[test]
    fn valid_choice_answer() {
        let kind = QuestionKind::Choice {
            options: vec![opt("a", "A"), opt("b", "B")],
            default: None,
        };
        let errs = validate_answer(&kind, &[], &AnswerValue::Choice("a".into()));
        assert!(errs.is_empty());
    }

    #[test]
    fn invalid_choice_value() {
        let kind = QuestionKind::Choice {
            options: vec![opt("a", "A")],
            default: None,
        };
        let errs = validate_answer(&kind, &[], &AnswerValue::Choice("z".into()));
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("not a valid choice"));
    }

    #[test]
    fn type_mismatch() {
        let kind = QuestionKind::YesNo { default: None };
        let errs = validate_answer(&kind, &[], &AnswerValue::Choice("oops".into()));
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("does not match question kind"));
    }

    #[test]
    fn required_rule_empty_text() {
        let kind = QuestionKind::Text {
            placeholder: None,
            default: None,
        };
        let errs = validate_answer(&kind, &[ValidationRule::Required], &AnswerValue::Text(None));
        assert!(errs.iter().any(|e| e.contains("required")));
    }

    #[test]
    fn required_rule_blank_text() {
        let kind = QuestionKind::Text {
            placeholder: None,
            default: None,
        };
        let errs = validate_answer(
            &kind,
            &[ValidationRule::Required],
            &AnswerValue::Text(Some("   ".into())),
        );
        assert!(errs.iter().any(|e| e.contains("required")));
    }

    #[test]
    fn required_rule_present_text() {
        let kind = QuestionKind::Text {
            placeholder: None,
            default: None,
        };
        let errs = validate_answer(
            &kind,
            &[ValidationRule::Required],
            &AnswerValue::Text(Some("hello".into())),
        );
        assert!(errs.is_empty());
    }

    #[test]
    fn min_length_rule() {
        let kind = QuestionKind::Text {
            placeholder: None,
            default: None,
        };
        let errs = validate_answer(
            &kind,
            &[ValidationRule::MinLength(5)],
            &AnswerValue::Text(Some("hi".into())),
        );
        assert!(errs.iter().any(|e| e.contains("too short")));

        let errs2 = validate_answer(
            &kind,
            &[ValidationRule::MinLength(5)],
            &AnswerValue::Text(Some("hello".into())),
        );
        assert!(errs2.is_empty());
    }

    #[test]
    fn max_length_rule() {
        let kind = QuestionKind::Text {
            placeholder: None,
            default: None,
        };
        let errs = validate_answer(
            &kind,
            &[ValidationRule::MaxLength(3)],
            &AnswerValue::Text(Some("toolong".into())),
        );
        assert!(errs.iter().any(|e| e.contains("too long")));
    }

    #[test]
    fn pattern_rule_pass() {
        let kind = QuestionKind::Text {
            placeholder: None,
            default: None,
        };
        let errs = validate_answer(
            &kind,
            &[ValidationRule::Pattern(r"^\d+$".into())],
            &AnswerValue::Text(Some("123".into())),
        );
        assert!(errs.is_empty());
    }

    #[test]
    fn pattern_rule_fail() {
        let kind = QuestionKind::Text {
            placeholder: None,
            default: None,
        };
        let errs = validate_answer(
            &kind,
            &[ValidationRule::Pattern(r"^\d+$".into())],
            &AnswerValue::Text(Some("abc".into())),
        );
        assert!(errs.iter().any(|e| e.contains("does not match")));
    }

    #[test]
    fn range_rule() {
        let kind = QuestionKind::Number {
            min: None,
            max: None,
            default: None,
        };
        let errs = validate_answer(
            &kind,
            &[ValidationRule::Range {
                min: 1.0,
                max: 10.0,
            }],
            &AnswerValue::Number(15.0),
        );
        assert!(errs.iter().any(|e| e.contains("outside range")));

        let errs2 = validate_answer(
            &kind,
            &[ValidationRule::Range {
                min: 1.0,
                max: 10.0,
            }],
            &AnswerValue::Number(5.0),
        );
        assert!(errs2.is_empty());
    }

    #[test]
    fn number_kind_bounds() {
        let kind = QuestionKind::Number {
            min: Some(0.0),
            max: Some(100.0),
            default: None,
        };
        let errs = validate_answer(&kind, &[], &AnswerValue::Number(-1.0));
        assert!(errs.iter().any(|e| e.contains("below minimum")));

        let errs2 = validate_answer(&kind, &[], &AnswerValue::Number(200.0));
        assert!(errs2.iter().any(|e| e.contains("above maximum")));

        let errs3 = validate_answer(&kind, &[], &AnswerValue::Number(50.0));
        assert!(errs3.is_empty());
    }

    #[test]
    fn multi_select_valid() {
        let kind = QuestionKind::MultiSelect {
            options: vec![opt("a", "A"), opt("b", "B")],
            default: None,
        };
        let errs = validate_answer(
            &kind,
            &[],
            &AnswerValue::MultiSelect(vec!["a".into(), "b".into()]),
        );
        assert!(errs.is_empty());
    }

    #[test]
    fn multi_select_invalid_option() {
        let kind = QuestionKind::MultiSelect {
            options: vec![opt("a", "A")],
            default: None,
        };
        let errs = validate_answer(
            &kind,
            &[],
            &AnswerValue::MultiSelect(vec!["a".into(), "z".into()]),
        );
        assert!(errs.iter().any(|e| e.contains("'z'")));
    }

    #[test]
    fn required_multi_select_empty() {
        let kind = QuestionKind::MultiSelect {
            options: vec![opt("a", "A")],
            default: None,
        };
        let errs = validate_answer(
            &kind,
            &[ValidationRule::Required],
            &AnswerValue::MultiSelect(vec![]),
        );
        assert!(errs.iter().any(|e| e.contains("required")));
    }

    // -------- validate_questionnaire_schema tests --------

    fn make_simple_questionnaire(questions: Vec<Question>) -> Questionnaire {
        Questionnaire {
            id: QuestionnaireId::new("test"),
            title: "Test".into(),
            description: "A test questionnaire".into(),
            questions,
        }
    }

    fn choice_question(id: &str, options: Vec<(&str, &str)>, default: Option<&str>) -> Question {
        Question {
            id: QuestionId::new(id),
            label: id.to_string(),
            help_text: None,
            kind: QuestionKind::Choice {
                options: options
                    .into_iter()
                    .map(|(v, l)| ChoiceOption {
                        value: v.into(),
                        label: l.into(),
                        description: None,
                    })
                    .collect(),
                default: default.map(|s| s.into()),
            },
            required: false,
            validation: vec![],
            condition: None,
        }
    }

    #[test]
    fn valid_questionnaire_schema() {
        let q = make_simple_questionnaire(vec![
            choice_question("q1", vec![("a", "A"), ("b", "B")], Some("a")),
            choice_question("q2", vec![("x", "X")], None),
        ]);
        assert!(validate_questionnaire_schema(&q).is_ok());
    }

    #[test]
    fn rejects_duplicate_ids() {
        let q = make_simple_questionnaire(vec![
            choice_question("dup", vec![("a", "A")], None),
            choice_question("dup", vec![("b", "B")], None),
        ]);
        let errs = validate_questionnaire_schema(&q).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("duplicate question id")));
    }

    #[test]
    fn rejects_default_not_in_options() {
        let q = make_simple_questionnaire(vec![choice_question(
            "q1",
            vec![("a", "A"), ("b", "B")],
            Some("c"),
        )]);
        let errs = validate_questionnaire_schema(&q).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("default 'c'")));
    }

    #[test]
    fn rejects_empty_choice_options() {
        let q = make_simple_questionnaire(vec![choice_question("q1", vec![], None)]);
        let errs = validate_questionnaire_schema(&q).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("at least one option")));
    }

    #[test]
    fn rejects_duplicate_choice_values() {
        let q = make_simple_questionnaire(vec![choice_question(
            "q1",
            vec![("a", "A"), ("a", "Also A")],
            None,
        )]);
        let errs = validate_questionnaire_schema(&q).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("duplicate option value")));
    }

    #[test]
    fn rejects_number_min_gt_max() {
        let q = make_simple_questionnaire(vec![Question {
            id: QuestionId::new("num"),
            label: "Number".into(),
            help_text: None,
            kind: QuestionKind::Number {
                min: Some(100.0),
                max: Some(10.0),
                default: None,
            },
            required: false,
            validation: vec![],
            condition: None,
        }]);
        let errs = validate_questionnaire_schema(&q).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.contains("min") && e.contains("greater than max"))
        );
    }

    #[test]
    fn rejects_multi_select_default_not_in_options() {
        let q = make_simple_questionnaire(vec![Question {
            id: QuestionId::new("ms"),
            label: "Multi".into(),
            help_text: None,
            kind: QuestionKind::MultiSelect {
                options: vec![opt("a", "A"), opt("b", "B")],
                default: Some(vec!["a".into(), "z".into()]),
            },
            required: false,
            validation: vec![],
            condition: None,
        }]);
        let errs = validate_questionnaire_schema(&q).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("'z'")));
    }

    #[test]
    fn rejects_number_default_out_of_bounds() {
        let q = make_simple_questionnaire(vec![Question {
            id: QuestionId::new("num"),
            label: "Number".into(),
            help_text: None,
            kind: QuestionKind::Number {
                min: Some(0.0),
                max: Some(10.0),
                default: Some(20.0),
            },
            required: false,
            validation: vec![],
            condition: None,
        }]);
        let errs = validate_questionnaire_schema(&q).unwrap_err();
        assert!(
            errs.iter()
                .any(|e| e.contains("default") && e.contains("above max"))
        );
    }
}
