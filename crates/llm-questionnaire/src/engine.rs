use crate::answer::{AnswerMap, AnswerValue};
use crate::schema::{Question, QuestionKind, Questionnaire, Section};
use crate::validate::{validate_answer, validate_questionnaire_schema};

/// Drives a single run through a [`Questionnaire`], tracking the current
/// position and accumulated answers.
#[derive(Debug)]
pub struct QuestionnaireRun {
    questionnaire: Questionnaire,
    answers: AnswerMap,
    current_index: usize,
}

impl QuestionnaireRun {
    /// Create a new run, validating the questionnaire schema up front.
    pub fn new(questionnaire: Questionnaire) -> Result<Self, Vec<String>> {
        validate_questionnaire_schema(&questionnaire)?;
        Ok(Self {
            questionnaire,
            answers: AnswerMap::new(),
            current_index: 0,
        })
    }

    /// Returns a reference to the underlying questionnaire.
    pub fn questionnaire(&self) -> &Questionnaire {
        &self.questionnaire
    }

    /// Returns the next visible question (evaluating conditions), or `None` if
    /// the questionnaire is complete.
    ///
    /// Questions whose condition evaluates to `false` are automatically skipped.
    pub fn next_question(&self) -> Option<&Question> {
        let questions = &self.questionnaire.questions;
        let mut idx = self.current_index;
        while idx < questions.len() {
            let q = &questions[idx];
            let visible = match &q.condition {
                Some(cond) => cond.evaluate(&self.answers),
                None => true,
            };
            if visible {
                return Some(q);
            }
            idx += 1;
        }
        None
    }

    /// Submit an answer for the current question.
    ///
    /// The answer is validated against the question's kind, explicit validation
    /// rules, and required flag. On success the internal index advances; on
    /// failure a vec of error messages is returned.
    pub fn submit_answer(&mut self, answer: AnswerValue) -> Result<(), Vec<String>> {
        let question = match self.resolve_current_question() {
            Some(q) => q,
            None => return Err(vec!["no current question to answer".into()]),
        };

        // Combine explicit rules with the implicit required check.
        let mut rules = question.validation.clone();
        if question.required
            && !rules
                .iter()
                .any(|r| matches!(r, crate::validate::ValidationRule::Required))
        {
            rules.insert(0, crate::validate::ValidationRule::Required);
        }

        let errors = validate_answer(&question.kind, &rules, &answer);
        if !errors.is_empty() {
            return Err(errors);
        }

        let qid = question.id.clone();
        // Insert the answer first, then advance past the answered question
        // (and any subsequent invisible ones) so that condition evaluation
        // during advance sees the newly recorded answer.
        self.answers.insert(qid, answer);
        self.advance_past_current();
        Ok(())
    }

    /// Advance past an info item without collecting an answer.
    ///
    /// Returns `Err` if the current visible question is not an info item.
    pub fn advance_info(&mut self) -> Result<(), Vec<String>> {
        let question = match self.resolve_current_question() {
            Some(q) => q,
            None => return Err(vec!["no current question to advance".into()]),
        };
        if !matches!(question.kind, QuestionKind::Info { .. }) {
            return Err(vec!["current question is not an info item".into()]);
        }
        self.advance_past_current();
        Ok(())
    }

    /// The accumulated answers so far.
    pub fn answers(&self) -> &AnswerMap {
        &self.answers
    }

    /// Consume the run and return the final answers.
    pub fn into_answers(self) -> AnswerMap {
        self.answers
    }

    /// True when every visible question has been answered.
    pub fn is_complete(&self) -> bool {
        self.next_question().is_none()
    }

    /// The raw index into the questions vec (exposed for diagnostics).
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Returns the section that contains the current question, or `None` if
    /// sections are not defined or the questionnaire is complete.
    pub fn current_section(&self) -> Option<&Section> {
        self.resolve_current_visible_index()
            .and_then(|idx| self.questionnaire.section_of_index(idx))
    }

    // --- internal helpers ---

    /// Resolve the current visible question (same as `next_question` but returns
    /// the index as well).
    fn resolve_current_visible_index(&self) -> Option<usize> {
        let questions = &self.questionnaire.questions;
        let mut idx = self.current_index;
        while idx < questions.len() {
            let q = &questions[idx];
            let visible = match &q.condition {
                Some(cond) => cond.evaluate(&self.answers),
                None => true,
            };
            if visible {
                return Some(idx);
            }
            idx += 1;
        }
        None
    }

    fn resolve_current_question(&self) -> Option<&Question> {
        self.resolve_current_visible_index()
            .map(|idx| &self.questionnaire.questions[idx])
    }

    fn advance_past_current(&mut self) {
        if let Some(idx) = self.resolve_current_visible_index() {
            self.current_index = idx + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ids::{QuestionId, QuestionnaireId};

    use super::*;
    use crate::condition::ConditionExpr;
    use crate::schema::{ChoiceOption, Question, QuestionKind, Questionnaire};
    use crate::validate::ValidationRule;

    fn yes_no_question(id: &str, required: bool) -> Question {
        Question {
            id: QuestionId::new(id),
            label: id.into(),
            help_text: None,
            kind: QuestionKind::YesNo { default: None },
            required,
            validation: vec![],
            condition: None,
        }
    }

    fn choice_question(id: &str, opts: &[&str], default: Option<&str>) -> Question {
        Question {
            id: QuestionId::new(id),
            label: id.into(),
            help_text: None,
            kind: QuestionKind::Choice {
                options: opts
                    .iter()
                    .map(|v| ChoiceOption {
                        value: (*v).into(),
                        label: v.to_uppercase(),
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

    fn text_question(id: &str) -> Question {
        Question {
            id: QuestionId::new(id),
            label: id.into(),
            help_text: None,
            kind: QuestionKind::Text {
                placeholder: None,
                default: None,
            },
            required: false,
            validation: vec![],
            condition: None,
        }
    }

    fn make_questionnaire(questions: Vec<Question>) -> Questionnaire {
        Questionnaire {
            id: QuestionnaireId::new("test"),
            title: "Test".into(),
            description: "desc".into(),
            sections: vec![],
            questions,
        }
    }

    // --- basic flow ---

    #[test]
    fn sequential_answering() {
        let q = make_questionnaire(vec![
            yes_no_question("q1", false),
            yes_no_question("q2", false),
        ]);
        let mut run = QuestionnaireRun::new(q).unwrap();

        assert!(!run.is_complete());
        assert_eq!(run.next_question().unwrap().id.as_str(), "q1");

        run.submit_answer(AnswerValue::YesNo(true)).unwrap();
        assert_eq!(run.next_question().unwrap().id.as_str(), "q2");

        run.submit_answer(AnswerValue::YesNo(false)).unwrap();
        assert!(run.is_complete());
        assert!(run.next_question().is_none());
    }

    #[test]
    fn answers_are_recorded() {
        let q = make_questionnaire(vec![choice_question(
            "color",
            &["red", "blue"],
            Some("red"),
        )]);
        let mut run = QuestionnaireRun::new(q).unwrap();
        run.submit_answer(AnswerValue::Choice("blue".into()))
            .unwrap();
        assert_eq!(
            run.answers().choice(&QuestionId::new("color")),
            Some("blue")
        );
    }

    // --- validation rejection ---

    #[test]
    fn rejects_invalid_choice() {
        let q = make_questionnaire(vec![choice_question("q1", &["a", "b"], None)]);
        let mut run = QuestionnaireRun::new(q).unwrap();
        let errs = run
            .submit_answer(AnswerValue::Choice("z".into()))
            .unwrap_err();
        assert!(errs.iter().any(|e| e.contains("not a valid choice")));
        // Question should still be current (not advanced).
        assert_eq!(run.next_question().unwrap().id.as_str(), "q1");
    }

    #[test]
    fn rejects_type_mismatch() {
        let q = make_questionnaire(vec![yes_no_question("q1", false)]);
        let mut run = QuestionnaireRun::new(q).unwrap();
        let errs = run
            .submit_answer(AnswerValue::Choice("oops".into()))
            .unwrap_err();
        assert!(!errs.is_empty());
    }

    #[test]
    fn required_field_enforced() {
        let q = make_questionnaire(vec![{
            let mut question = text_question("name");
            question.required = true;
            question
        }]);
        let mut run = QuestionnaireRun::new(q).unwrap();
        let errs = run.submit_answer(AnswerValue::Text(None)).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("required")));
    }

    // --- conditional skip ---

    #[test]
    fn conditional_question_skipped() {
        let q = make_questionnaire(vec![
            yes_no_question("use_advanced", false),
            {
                let mut question = text_question("advanced_setting");
                question.condition = Some(ConditionExpr::Equals {
                    question_id: QuestionId::new("use_advanced"),
                    value: serde_json::Value::Bool(true),
                });
                question
            },
            yes_no_question("done", false),
        ]);
        let mut run = QuestionnaireRun::new(q).unwrap();

        // Answer "no" to use_advanced.
        assert_eq!(run.next_question().unwrap().id.as_str(), "use_advanced");
        run.submit_answer(AnswerValue::YesNo(false)).unwrap();

        // advanced_setting should be skipped; next should be "done".
        assert_eq!(run.next_question().unwrap().id.as_str(), "done");
        run.submit_answer(AnswerValue::YesNo(true)).unwrap();
        assert!(run.is_complete());
    }

    #[test]
    fn conditional_question_shown_when_condition_met() {
        let q = make_questionnaire(vec![
            yes_no_question("use_advanced", false),
            {
                let mut question = text_question("advanced_setting");
                question.condition = Some(ConditionExpr::Equals {
                    question_id: QuestionId::new("use_advanced"),
                    value: serde_json::Value::Bool(true),
                });
                question
            },
            yes_no_question("done", false),
        ]);
        let mut run = QuestionnaireRun::new(q).unwrap();

        // Answer "yes" to use_advanced.
        run.submit_answer(AnswerValue::YesNo(true)).unwrap();

        // advanced_setting should be visible.
        assert_eq!(run.next_question().unwrap().id.as_str(), "advanced_setting");
        run.submit_answer(AnswerValue::Text(Some("custom".into())))
            .unwrap();

        assert_eq!(run.next_question().unwrap().id.as_str(), "done");
    }

    // --- schema validation at construction ---

    #[test]
    fn new_rejects_invalid_schema() {
        let q = Questionnaire {
            id: QuestionnaireId::new("bad"),
            title: "Bad".into(),
            description: "desc".into(),
            sections: vec![],
            questions: vec![
                choice_question("dup", &["a"], None),
                choice_question("dup", &["b"], None),
            ],
        };
        let errs = QuestionnaireRun::new(q).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("duplicate")));
    }

    // --- submit when complete ---

    #[test]
    fn submit_after_complete_returns_error() {
        let q = make_questionnaire(vec![yes_no_question("q1", false)]);
        let mut run = QuestionnaireRun::new(q).unwrap();
        run.submit_answer(AnswerValue::YesNo(true)).unwrap();
        assert!(run.is_complete());

        let errs = run.submit_answer(AnswerValue::YesNo(false)).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("no current question")));
    }

    // --- explicit validation rules in engine ---

    #[test]
    fn validation_rules_applied_during_submit() {
        let q = make_questionnaire(vec![{
            let mut question = text_question("name");
            question.validation = vec![ValidationRule::MinLength(3)];
            question
        }]);
        let mut run = QuestionnaireRun::new(q).unwrap();
        let errs = run
            .submit_answer(AnswerValue::Text(Some("ab".into())))
            .unwrap_err();
        assert!(errs.iter().any(|e| e.contains("too short")));
    }

    // --- empty questionnaire ---

    #[test]
    fn empty_questionnaire_is_immediately_complete() {
        let q = make_questionnaire(vec![]);
        let run = QuestionnaireRun::new(q).unwrap();
        assert!(run.is_complete());
        assert!(run.next_question().is_none());
    }

    // --- all questions conditional and none visible ---

    #[test]
    fn all_conditional_invisible_is_complete() {
        // "hidden" depends on "gate" being answered, but "gate" itself is
        // also conditional (on an unanswered question), so neither is
        // visible and the run is immediately complete.
        let q = make_questionnaire(vec![
            {
                let mut gate = yes_no_question("gate", false);
                gate.condition = Some(ConditionExpr::Equals {
                    question_id: QuestionId::new("hidden"),
                    value: serde_json::Value::String("trigger".into()),
                });
                gate
            },
            {
                let mut question = text_question("hidden");
                question.condition = Some(ConditionExpr::Answered {
                    question_id: QuestionId::new("gate"),
                });
                question
            },
        ]);
        let run = QuestionnaireRun::new(q).unwrap();
        assert!(run.is_complete());
    }
}
