use serde::{Deserialize, Serialize};

use crate::ids::QuestionId;

use crate::answer::AnswerMap;

/// A boolean expression evaluated against current answers to determine
/// whether a question should be shown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ConditionExpr {
    /// True when the answer for `question_id` equals `value`.
    Equals {
        question_id: QuestionId,
        value: serde_json::Value,
    },
    /// True when the answer for `question_id` does **not** equal `value`.
    NotEquals {
        question_id: QuestionId,
        value: serde_json::Value,
    },
    /// True when `question_id` has been answered (present in the map).
    Answered { question_id: QuestionId },
    /// True when **all** inner conditions are true.
    And(Vec<ConditionExpr>),
    /// True when **any** inner condition is true.
    Or(Vec<ConditionExpr>),
    /// Logical negation of the inner condition.
    Not(Box<ConditionExpr>),
}

impl ConditionExpr {
    /// Evaluate this condition against the supplied answer map.
    pub fn evaluate(&self, answers: &AnswerMap) -> bool {
        match self {
            ConditionExpr::Equals { question_id, value } => answers
                .to_json_value(question_id)
                .map(|v| &v == value)
                .unwrap_or(false),

            ConditionExpr::NotEquals { question_id, value } => answers
                .to_json_value(question_id)
                .map(|v| &v != value)
                .unwrap_or(true),

            ConditionExpr::Answered { question_id } => answers.contains(question_id),

            ConditionExpr::And(exprs) => exprs.iter().all(|e| e.evaluate(answers)),

            ConditionExpr::Or(exprs) => exprs.iter().any(|e| e.evaluate(answers)),

            ConditionExpr::Not(inner) => !inner.evaluate(answers),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_answers() -> AnswerMap {
        use crate::answer::AnswerValue;
        let mut map = AnswerMap::new();
        map.insert(QuestionId::new("color"), AnswerValue::Choice("red".into()));
        map.insert(QuestionId::new("agree"), AnswerValue::YesNo(true));
        map.insert(
            QuestionId::new("name"),
            AnswerValue::Text(Some("Alice".into())),
        );
        map.insert(QuestionId::new("count"), AnswerValue::Number(42.0));
        map
    }

    #[test]
    fn equals_matches_choice() {
        let answers = make_answers();
        let cond = ConditionExpr::Equals {
            question_id: QuestionId::new("color"),
            value: serde_json::Value::String("red".into()),
        };
        assert!(cond.evaluate(&answers));
    }

    #[test]
    fn equals_does_not_match_wrong_value() {
        let answers = make_answers();
        let cond = ConditionExpr::Equals {
            question_id: QuestionId::new("color"),
            value: serde_json::Value::String("blue".into()),
        };
        assert!(!cond.evaluate(&answers));
    }

    #[test]
    fn equals_missing_question_is_false() {
        let answers = make_answers();
        let cond = ConditionExpr::Equals {
            question_id: QuestionId::new("missing"),
            value: serde_json::Value::String("x".into()),
        };
        assert!(!cond.evaluate(&answers));
    }

    #[test]
    fn not_equals_works() {
        let answers = make_answers();
        let cond = ConditionExpr::NotEquals {
            question_id: QuestionId::new("color"),
            value: serde_json::Value::String("blue".into()),
        };
        assert!(cond.evaluate(&answers));

        let cond2 = ConditionExpr::NotEquals {
            question_id: QuestionId::new("color"),
            value: serde_json::Value::String("red".into()),
        };
        assert!(!cond2.evaluate(&answers));
    }

    #[test]
    fn not_equals_missing_question_is_true() {
        let answers = make_answers();
        let cond = ConditionExpr::NotEquals {
            question_id: QuestionId::new("missing"),
            value: serde_json::Value::String("x".into()),
        };
        assert!(cond.evaluate(&answers));
    }

    #[test]
    fn answered_present() {
        let answers = make_answers();
        let cond = ConditionExpr::Answered {
            question_id: QuestionId::new("agree"),
        };
        assert!(cond.evaluate(&answers));
    }

    #[test]
    fn answered_missing() {
        let answers = make_answers();
        let cond = ConditionExpr::Answered {
            question_id: QuestionId::new("missing"),
        };
        assert!(!cond.evaluate(&answers));
    }

    #[test]
    fn and_all_true() {
        let answers = make_answers();
        let cond = ConditionExpr::And(vec![
            ConditionExpr::Answered {
                question_id: QuestionId::new("color"),
            },
            ConditionExpr::Answered {
                question_id: QuestionId::new("agree"),
            },
        ]);
        assert!(cond.evaluate(&answers));
    }

    #[test]
    fn and_one_false() {
        let answers = make_answers();
        let cond = ConditionExpr::And(vec![
            ConditionExpr::Answered {
                question_id: QuestionId::new("color"),
            },
            ConditionExpr::Answered {
                question_id: QuestionId::new("nope"),
            },
        ]);
        assert!(!cond.evaluate(&answers));
    }

    #[test]
    fn and_empty_is_true() {
        let answers = make_answers();
        let cond = ConditionExpr::And(vec![]);
        assert!(cond.evaluate(&answers));
    }

    #[test]
    fn or_one_true() {
        let answers = make_answers();
        let cond = ConditionExpr::Or(vec![
            ConditionExpr::Answered {
                question_id: QuestionId::new("nope"),
            },
            ConditionExpr::Answered {
                question_id: QuestionId::new("color"),
            },
        ]);
        assert!(cond.evaluate(&answers));
    }

    #[test]
    fn or_none_true() {
        let answers = make_answers();
        let cond = ConditionExpr::Or(vec![
            ConditionExpr::Answered {
                question_id: QuestionId::new("nope1"),
            },
            ConditionExpr::Answered {
                question_id: QuestionId::new("nope2"),
            },
        ]);
        assert!(!cond.evaluate(&answers));
    }

    #[test]
    fn or_empty_is_false() {
        let answers = make_answers();
        let cond = ConditionExpr::Or(vec![]);
        assert!(!cond.evaluate(&answers));
    }

    #[test]
    fn not_inverts() {
        let answers = make_answers();
        let cond = ConditionExpr::Not(Box::new(ConditionExpr::Answered {
            question_id: QuestionId::new("color"),
        }));
        assert!(!cond.evaluate(&answers));

        let cond2 = ConditionExpr::Not(Box::new(ConditionExpr::Answered {
            question_id: QuestionId::new("nope"),
        }));
        assert!(cond2.evaluate(&answers));
    }

    #[test]
    fn equals_yes_no_bool() {
        let answers = make_answers();
        let cond = ConditionExpr::Equals {
            question_id: QuestionId::new("agree"),
            value: serde_json::Value::Bool(true),
        };
        assert!(cond.evaluate(&answers));

        let cond2 = ConditionExpr::Equals {
            question_id: QuestionId::new("agree"),
            value: serde_json::Value::Bool(false),
        };
        assert!(!cond2.evaluate(&answers));
    }

    #[test]
    fn equals_number() {
        let answers = make_answers();
        let cond = ConditionExpr::Equals {
            question_id: QuestionId::new("count"),
            value: serde_json::json!(42.0),
        };
        assert!(cond.evaluate(&answers));
    }
}
