pub mod answer;
pub mod builder;
pub mod condition;
pub mod engine;
pub mod ids;
pub mod schema;
pub mod tool;
pub mod validate;

pub use answer::{AnswerMap, AnswerValue};
pub use builder::{QuestionConfig, QuestionnaireBuilder};
pub use condition::ConditionExpr;
pub use engine::QuestionnaireRun;
pub use ids::{QuestionId, QuestionnaireId};
pub use schema::{ChoiceOption, Question, QuestionKind, Questionnaire};
pub use tool::{
    AskQuestionsInput, AskQuestionsOutput, AskQuestionsTool, MAX_QUESTIONS_PER_CALL, OptionInput,
    QuestionHandler, QuestionInput, QuestionType, SkipCondition,
};
pub use validate::{ValidationRule, validate_answer, validate_questionnaire_schema};

// Re-export schemars so downstream crates can call `schema_for!(Questionnaire)`
// without adding schemars as a direct dependency. Note: deriving `JsonSchema`
// on your own types still requires a direct schemars dependency.
pub use schemars::{self, JsonSchema, schema_for};
