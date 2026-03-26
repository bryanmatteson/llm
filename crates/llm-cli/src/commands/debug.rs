use clap::{Args, Subcommand};

use llm_core::{FrameworkError, Result};

use crate::bootstrap::AppContext;

#[derive(Args)]
pub struct DebugArgs {
    #[command(subcommand)]
    pub command: DebugCommands,
}

#[derive(Subcommand)]
pub enum DebugCommands {
    /// List registered providers and their capabilities.
    Providers,

    /// List models for a provider (placeholder).
    Models {
        /// Provider identifier.
        provider: String,
    },

    /// Print the current framework configuration.
    Config,
}

pub async fn run_debug(args: DebugArgs, ctx: &AppContext) -> Result<()> {
    match args.command {
        DebugCommands::Providers => providers(ctx),
        DebugCommands::Models { provider } => models(&provider, ctx).await,
        DebugCommands::Config => config(ctx),
    }
}

fn providers(ctx: &AppContext) -> Result<()> {
    if ctx.provider_descriptors.is_empty() {
        println!("No providers registered.");
        return Ok(());
    }

    for desc in &ctx.provider_descriptors {
        println!("Provider: {} ({})", desc.display_name, desc.id);
        println!("  Default model: {}", desc.default_model);
        println!(
            "  Capabilities:  {}",
            desc.capabilities
                .iter()
                .map(|c| format!("{c:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        if !desc.metadata.is_empty() {
            println!("  Metadata:");
            for (k, v) in &desc.metadata {
                println!("    {k}: {v}");
            }
        }
        println!();
    }

    Ok(())
}

async fn models(provider: &str, ctx: &AppContext) -> Result<()> {
    let _desc = ctx
        .provider_descriptor(provider)
        .ok_or_else(|| FrameworkError::config(format!("unknown provider: {provider}")))?;

    // To list models we'd need an authenticated client. For now, show a
    // placeholder message.
    let session = ctx
        .credential_store
        .get_auth_session(&llm_core::ProviderId::new(provider))
        .await?;

    if session.is_none() {
        eprintln!(
            "Not logged in to {provider}. Run `llmctl auth login {provider}` to list models."
        );
        return Ok(());
    }

    let auth_session = session.unwrap();
    let model_id = ctx
        .provider_descriptor(provider)
        .map(|d| d.default_model.clone())
        .unwrap_or_else(|| llm_core::ModelId::new("gpt-4o-mini"));

    let client = llm_provider_openai::OpenAiClient::new(
        auth_session,
        model_id,
        llm_provider_openai::API_BASE,
    );

    use llm_provider_api::LlmProviderClient;
    let models = client.list_models().await?;

    if models.is_empty() {
        println!("No models returned by {provider}.");
        return Ok(());
    }

    println!("{:<32} DISPLAY NAME", "MODEL ID");
    println!("{}", "-".repeat(56));
    for m in &models {
        println!("{:<32} {}", m.id, m.display_name);
    }

    Ok(())
}

fn config(ctx: &AppContext) -> Result<()> {
    println!("LLM Framework Configuration");
    println!("===========================");
    println!();
    println!(
        "Registered providers: {}",
        ctx.provider_descriptors.len()
    );
    for desc in &ctx.provider_descriptors {
        println!("  - {} ({})", desc.display_name, desc.id);
    }
    println!();
    println!(
        "Registered tools:     {}",
        ctx.tool_registry.all_descriptors().len()
    );
    for desc in &ctx.tool_registry.all_descriptors() {
        println!("  - {} ({})", desc.display_name, desc.id);
    }
    println!();
    println!(
        "Auth providers:       {}",
        ctx.auth_providers.len()
    );
    for ap in &ctx.auth_providers {
        println!("  - {}", ap.provider_id());
    }

    Ok(())
}
