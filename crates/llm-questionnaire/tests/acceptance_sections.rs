//! Acceptance tests for questionnaire sections and info question type.
//!
//! These tests start RED — the Section type, Info question kind, and section
//! builder API do not exist yet. Phase 1 of the interactive-config plan
//! turns them green.
//!
//! # Phase 1: Questionnaire extensions

use llm_questionnaire::{
    AnswerValue, QuestionId, QuestionnaireBuilder, SectionId,
    engine::QuestionnaireRun,
    schema::QuestionKind,
};

// ---------------------------------------------------------------------------
// Section builder
// ---------------------------------------------------------------------------

/// Sections group questions under a titled header. The builder API creates
/// sections via a closure that receives a SectionBuilder.
///
/// # Phase 1
#[test]
fn test_section_builder_groups_questions() {
    let q = QuestionnaireBuilder::new("setup", "Setup")
        .section("embedding", "Embedding", |s| {
            s.description("Choose your embedding backend")
                .choice("backend", "Backend?", &["onnx", "ollama", "lmstudio"])
                .yes_no("split_pools", "Separate pools for code vs prose?")
        })
        .section("answer", "Answer Generation", |s| {
            s.description("Configure LLM answer provider")
                .choice("provider", "Provider?", &["anthropic", "openai", "template"])
        })
        .build();

    assert_eq!(q.sections.len(), 2);
    assert_eq!(q.sections[0].id, SectionId::new("embedding"));
    assert_eq!(q.sections[0].title, "Embedding");
    assert_eq!(q.sections[0].questions.len(), 2);
    assert_eq!(q.sections[1].id, SectionId::new("answer"));
    assert_eq!(q.sections[1].questions.len(), 1);

    // Flat accessor should return all 3 questions
    assert_eq!(q.questions.len(), 3);
}

/// Questions added without a section call go into an implicit default section.
/// This preserves backward compatibility with existing callers like stag's
/// build_init_questionnaire().
///
/// # Phase 1
#[test]
fn test_backward_compat_flat_questionnaire() {
    let q = QuestionnaireBuilder::new("legacy", "Legacy")
        .choice("q1", "Question 1?", &["a", "b"])
        .yes_no("q2", "Question 2?")
        .text("q3", "Question 3?")
        .build();

    // Should have one implicit section with all 3 questions
    assert_eq!(q.sections.len(), 1);
    assert!(q.sections[0].title.is_empty() || q.sections[0].id.as_str().is_empty());
    assert_eq!(q.sections[0].questions.len(), 3);
    assert_eq!(q.questions.len(), 3);
}

// ---------------------------------------------------------------------------
// Info question type
// ---------------------------------------------------------------------------

/// Info items are display-only — they appear in the questionnaire but produce
/// no entry in the AnswerMap.
///
/// # Phase 1
#[test]
fn test_info_question_not_in_answers() {
    let q = QuestionnaireBuilder::new("preview", "Preview")
        .section("config", "Config Preview", |s| {
            s.info(
                "preview",
                "Generated config:",
                "embedding backend=onnx {\n  pool default { ... }\n}",
            )
            .yes_no("confirm", "Write this config?")
        })
        .build();

    let mut run = QuestionnaireRun::new(q).expect("valid schema");

    // First visible item should be the info block (or it should be auto-skipped
    // and we land on the yes_no). Either way, info should not be in answers.
    // Walk through the questionnaire:
    while let Some(question) = run.next_question() {
        match &question.kind {
            QuestionKind::Info { .. } => {
                // Engine should auto-advance past info items
                run.advance_info().expect("info advance should succeed");
            }
            QuestionKind::YesNo { .. } => {
                run.submit_answer(AnswerValue::YesNo(true)).expect("valid answer");
            }
            _ => panic!("unexpected question kind"),
        }
    }

    let answers = run.into_answers();
    assert!(answers.yes_no(&QuestionId::new("confirm")).is_some());
    assert!(
        answers.choice(&QuestionId::new("preview")).is_none(),
        "info items should not appear in answers"
    );
}

// ---------------------------------------------------------------------------
// Cross-section conditions
// ---------------------------------------------------------------------------

/// A condition in section B can reference a question in section A. Question IDs
/// are globally unique, not scoped to sections.
///
/// # Phase 1
#[test]
fn test_conditions_across_sections() {
    let q = QuestionnaireBuilder::new("cross", "Cross-Section")
        .section("general", "General", |s| {
            s.choice("mode", "Mode?", &["simple", "advanced"])
        })
        .section("advanced_opts", "Advanced Options", |s| {
            s.text_with("custom_endpoint", "Custom endpoint URL?", |c| {
                c.show_if_equals("mode", "advanced")
            })
        })
        .build();

    // When mode=simple, the advanced question should be skipped
    let mut run = QuestionnaireRun::new(q.clone()).expect("valid schema");
    let first = run.next_question().expect("should have first question");
    assert_eq!(first.id, QuestionId::new("mode"));
    run.submit_answer(AnswerValue::Choice("simple".into())).unwrap();

    // Next question should be None (custom_endpoint is hidden)
    assert!(
        run.next_question().is_none(),
        "custom_endpoint should be hidden when mode=simple"
    );

    // When mode=advanced, the advanced question should appear
    let mut run2 = QuestionnaireRun::new(q).expect("valid schema");
    run2.submit_answer(AnswerValue::Choice("advanced".into())).unwrap();
    let next = run2.next_question().expect("custom_endpoint should be visible");
    assert_eq!(next.id, QuestionId::new("custom_endpoint"));
}

// ---------------------------------------------------------------------------
// Engine section walking
// ---------------------------------------------------------------------------

/// The engine presents questions in section order: all of section 1, then
/// all of section 2, etc.
///
/// # Phase 1
#[test]
fn test_engine_walks_sections_in_order() {
    let q = QuestionnaireBuilder::new("ordered", "Ordered")
        .section("first", "First Section", |s| {
            s.choice("q1", "Q1?", &["a", "b"])
                .yes_no("q2", "Q2?")
        })
        .section("second", "Second Section", |s| {
            s.text("q3", "Q3?")
                .choice("q4", "Q4?", &["x", "y"])
        })
        .build();

    let mut run = QuestionnaireRun::new(q).expect("valid schema");
    let mut order = Vec::new();

    while let Some(question) = run.next_question() {
        let id = question.id.as_str().to_string();
        match &question.kind {
            QuestionKind::Choice { options, .. } => {
                run.submit_answer(AnswerValue::Choice(options[0].value.clone())).unwrap();
            }
            QuestionKind::YesNo { .. } => {
                run.submit_answer(AnswerValue::YesNo(true)).unwrap();
            }
            QuestionKind::Text { .. } => {
                run.submit_answer(AnswerValue::Text(Some("test".into()))).unwrap();
            }
            _ => panic!("unexpected kind"),
        }
        order.push(id);
    }

    assert_eq!(order, vec!["q1", "q2", "q3", "q4"]);
}
