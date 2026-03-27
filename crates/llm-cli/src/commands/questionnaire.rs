use clap::{Args, Subcommand};

use llm_core::Result;
use llm_questionnaire::{
    ChoiceOption, Question, QuestionId, QuestionKind, Questionnaire, QuestionnaireId,
};

use crate::bootstrap::AppContext;
use llm_cli::render::questionnaire::run_terminal_questionnaire;

#[derive(Args)]
pub struct QuestionnaireArgs {
    #[command(subcommand)]
    pub command: QuestionnaireCommands,
}

#[derive(Subcommand)]
pub enum QuestionnaireCommands {
    /// Run a questionnaire interactively.
    Run {
        /// Questionnaire identifier.
        id: String,
    },
}

pub async fn run_questionnaire(args: QuestionnaireArgs, _ctx: &AppContext) -> Result<()> {
    match args.command {
        QuestionnaireCommands::Run { id } => run(&id).await,
    }
}

async fn run(id: &str) -> Result<()> {
    // For now, load a built-in example questionnaire.
    let questionnaire = example_questionnaire(id)?;

    eprintln!("Starting questionnaire: {}", questionnaire.title);
    eprintln!("{}", questionnaire.description);
    eprintln!();

    let answers = run_terminal_questionnaire(&questionnaire)?;

    eprintln!();
    eprintln!(
        "Questionnaire complete! Collected {} answers.",
        answers.len()
    );

    // Print the results as JSON.
    let json = serde_json::to_string_pretty(&answers).unwrap_or_else(|_| format!("{answers:?}"));
    println!("{json}");

    Ok(())
}

/// Return a built-in example questionnaire for development/testing.
fn example_questionnaire(id: &str) -> Result<Questionnaire> {
    match id {
        "setup" => Ok(Questionnaire {
            id: QuestionnaireId::new("setup"),
            title: "Provider Setup".into(),
            description: "Configure your LLM provider settings.".into(),
            questions: vec![
                Question {
                    id: QuestionId::new("provider"),
                    label: "Which LLM provider would you like to use?".into(),
                    help_text: Some("Select your primary provider.".into()),
                    kind: QuestionKind::Choice {
                        options: vec![
                            ChoiceOption {
                                value: "openai".into(),
                                label: "OpenAI".into(),
                                description: None,
                            },
                            ChoiceOption {
                                value: "anthropic".into(),
                                label: "Anthropic".into(),
                                description: None,
                            },
                            ChoiceOption {
                                value: "local".into(),
                                label: "Local (Ollama)".into(),
                                description: None,
                            },
                        ],
                        default: Some("openai".into()),
                    },
                    required: true,
                    validation: vec![],
                    condition: None,
                },
                Question {
                    id: QuestionId::new("enable_tools"),
                    label: "Enable tool calling?".into(),
                    help_text: None,
                    kind: QuestionKind::YesNo {
                        default: Some(true),
                    },
                    required: true,
                    validation: vec![],
                    condition: None,
                },
                Question {
                    id: QuestionId::new("max_tokens"),
                    label: "Maximum tokens per response?".into(),
                    help_text: Some("Set 0 for provider default.".into()),
                    kind: QuestionKind::Number {
                        min: Some(0.0),
                        max: Some(128000.0),
                        default: Some(4096.0),
                    },
                    required: false,
                    validation: vec![],
                    condition: None,
                },
                Question {
                    id: QuestionId::new("system_prompt"),
                    label: "Custom system prompt (leave blank for default):".into(),
                    help_text: None,
                    kind: QuestionKind::Text {
                        placeholder: Some("You are a helpful assistant.".into()),
                        default: None,
                    },
                    required: false,
                    validation: vec![],
                    condition: None,
                },
            ],
        }),
        _ => Err(llm_core::FrameworkError::questionnaire(format!(
            "unknown questionnaire: {id}. Available: setup"
        ))),
    }
}
