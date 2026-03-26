mod bootstrap;
mod commands;

use clap::{Parser, Subcommand};

use crate::commands::auth::AuthArgs;
use crate::commands::debug::DebugArgs;
use crate::commands::questionnaire::QuestionnaireArgs;
use crate::commands::session::SessionArgs;
use crate::commands::tools::ToolsArgs;

/// Top-level CLI definition.
#[derive(Parser)]
#[command(name = "llmctl", about = "LLM Framework CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Manage authentication with LLM providers.
    Auth(AuthArgs),
    /// Create and manage chat sessions.
    Session(SessionArgs),
    /// Inspect registered tools.
    Tools(ToolsArgs),
    /// Run interactive questionnaires.
    Questionnaire(QuestionnaireArgs),
    /// Debug and inspect framework internals.
    Debug(DebugArgs),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let ctx = match bootstrap::build_default_context() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("error: failed to initialise: {e}");
            std::process::exit(1);
        }
    };

    let result = match cli.command {
        Commands::Auth(args) => commands::auth::run_auth(args, &ctx).await,
        Commands::Session(args) => commands::session::run_session(args, &ctx).await,
        Commands::Tools(args) => commands::tools::run_tools(args, &ctx).await,
        Commands::Questionnaire(args) => {
            commands::questionnaire::run_questionnaire(args, &ctx).await
        }
        Commands::Debug(args) => commands::debug::run_debug(args, &ctx).await,
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
