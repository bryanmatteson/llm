use std::io::{self, BufRead, Write};

use clap::{Args, Subcommand};

use llm_core::{FrameworkError, ModelId, ProviderId, Result};
use llm_provider_openai::{OpenAiClient, OpenAiToolFormat};
use llm_session::{SessionConfig, SessionHandle, TurnLoopContext, event_channel, run_turn_loop};

use crate::bootstrap::AppContext;
use llm_cli::approval::CliApprovalHandler;
use llm_cli::render::stream::render_session_events;

#[derive(Args)]
pub struct SessionArgs {
    #[command(subcommand)]
    pub command: SessionCommands,
}

#[derive(Subcommand)]
pub enum SessionCommands {
    /// Start a new chat session.
    New {
        /// Provider identifier (e.g. "openai").
        #[arg(short, long, default_value = "openai")]
        provider: String,

        /// Model override (uses provider default when omitted).
        #[arg(short, long)]
        model: Option<String>,
    },

    /// Resume an existing session or start a new one interactively.
    Chat {
        /// Optional session ID to resume.
        session_id: Option<String>,
    },

    /// List active sessions.
    List,
}

pub async fn run_session(args: SessionArgs, ctx: &AppContext) -> Result<()> {
    match args.command {
        SessionCommands::New { provider, model } => new_session(&provider, model, ctx).await,
        SessionCommands::Chat { session_id } => chat(session_id, ctx).await,
        SessionCommands::List => list_sessions(ctx).await,
    }
}

// ---------------------------------------------------------------------------
// New session
// ---------------------------------------------------------------------------

async fn new_session(provider: &str, model: Option<String>, ctx: &AppContext) -> Result<()> {
    let _desc = ctx
        .provider_descriptor(provider)
        .ok_or_else(|| FrameworkError::config(format!("unknown provider: {provider}")))?;

    let config = SessionConfig {
        provider_id: ProviderId::new(provider),
        model: model.map(ModelId::new),
        system_prompt: None,
        tool_policy: Default::default(),
        limits: Default::default(),
        metadata: Default::default(),
        provider_tools: Vec::new(),
        provider_request: Default::default(),
    };

    let handle = ctx.session_manager.create_session(config).await?;
    eprintln!("Created session: {}", handle.id);
    eprintln!("Type your messages below. Press Ctrl-D or type /quit to exit.");
    eprintln!();

    interactive_loop(handle, provider, ctx).await
}

// ---------------------------------------------------------------------------
// Chat (resume or new)
// ---------------------------------------------------------------------------

async fn chat(session_id: Option<String>, ctx: &AppContext) -> Result<()> {
    let handle = if let Some(id) = session_id {
        let sid = llm_core::SessionId::new(id.as_str());
        ctx.session_manager
            .get_session(&sid)
            .await?
            .ok_or_else(|| FrameworkError::session(format!("session not found: {id}")))?
    } else {
        // Create a fresh session on the default provider.
        let config = SessionConfig {
            provider_id: ProviderId::new("openai"),
            model: None,
            system_prompt: None,
            tool_policy: Default::default(),
            limits: Default::default(),
            metadata: Default::default(),
            provider_tools: Vec::new(),
            provider_request: Default::default(),
        };
        let h = ctx.session_manager.create_session(config).await?;
        eprintln!("Created session: {}", h.id);
        h
    };

    let provider = handle.config.provider_id.to_string();
    eprintln!("Type your messages below. Press Ctrl-D or type /quit to exit.");
    eprintln!();

    interactive_loop(handle, &provider, ctx).await
}

// ---------------------------------------------------------------------------
// Interactive loop
// ---------------------------------------------------------------------------

async fn interactive_loop(
    mut handle: SessionHandle,
    provider: &str,
    ctx: &AppContext,
) -> Result<()> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();

    loop {
        // Prompt
        eprint!("you> ");
        io::stderr().flush().ok();

        let mut line = String::new();
        let bytes = reader
            .read_line(&mut line)
            .map_err(|e| FrameworkError::session(format!("failed to read input: {e}")))?;

        // EOF
        if bytes == 0 {
            eprintln!();
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "/quit" || trimmed == "/exit" {
            break;
        }

        // Append the user message.
        handle.conversation.append_user(trimmed);

        // Build a provider client for this turn.
        let auth_session = ctx
            .credential_store
            .get_auth_session(&handle.config.provider_id)
            .await?
            .ok_or_else(|| {
                FrameworkError::auth(format!(
                    "not logged in to {provider}; run `llmctl auth login {provider}` first"
                ))
            })?;

        let model_id = handle.config.model.clone().unwrap_or_else(|| {
            ctx.provider_descriptor(provider)
                .map(|d| d.default_model.clone())
                .unwrap_or_else(|| ModelId::new("gpt-4o-mini"))
        });

        let client = OpenAiClient::from_session(auth_session, model_id);

        let adapter = OpenAiToolFormat;
        let approval = CliApprovalHandler;

        let (tx, mut rx) = event_channel();

        // Spawn a task to render events while the turn loop runs.
        let render_handle = tokio::spawn(async move {
            render_session_events(&mut rx).await;
        });

        let outcome = run_turn_loop(TurnLoopContext {
            session_id: &handle.id,
            client: &client,
            conversation: &mut handle.conversation,
            tool_registry: &ctx.tool_registry,
            tool_adapter: &adapter,
            config: &handle.config,
            approval_handler: &approval,
            event_tx: Some(&tx),
        })
        .await;

        // Drop sender so the render task can finish.
        drop(tx);
        let _ = render_handle.await;

        match outcome {
            Ok(_outcome) => {
                // The render task already printed the assistant response via
                // events; we just print a blank line separator.
                eprintln!();
            }
            Err(e) => {
                eprintln!("error: {e}");
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// List sessions
// ---------------------------------------------------------------------------

async fn list_sessions(ctx: &AppContext) -> Result<()> {
    let ids = ctx.session_manager.list_sessions().await?;

    if ids.is_empty() {
        println!("No active sessions.");
        return Ok(());
    }

    println!("{:<36} PROVIDER", "SESSION ID");
    println!("{}", "-".repeat(52));
    for id in &ids {
        // We only have the id from list; show what we can.
        println!("{id:<36} -");
    }

    Ok(())
}
