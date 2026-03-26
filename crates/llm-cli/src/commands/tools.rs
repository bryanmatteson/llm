use clap::{Args, Subcommand};

use llm_core::{FrameworkError, Result, ToolId};

use llm_cli::bootstrap::AppContext;

#[derive(Args)]
pub struct ToolsArgs {
    #[command(subcommand)]
    pub command: ToolsCommands,
}

#[derive(Subcommand)]
pub enum ToolsCommands {
    /// List all registered tools.
    List,

    /// Show details of a specific tool.
    Inspect {
        /// The tool identifier to inspect.
        tool_id: String,
    },
}

pub async fn run_tools(args: ToolsArgs, ctx: &AppContext) -> Result<()> {
    match args.command {
        ToolsCommands::List => list_tools(ctx),
        ToolsCommands::Inspect { tool_id } => inspect_tool(&tool_id, ctx),
    }
}

fn list_tools(ctx: &AppContext) -> Result<()> {
    let descriptors = ctx.tool_registry.all_descriptors();

    if descriptors.is_empty() {
        println!("No tools registered.");
        return Ok(());
    }

    println!("{:<20} {:<20} DESCRIPTION", "ID", "WIRE NAME");
    println!("{}", "-".repeat(72));
    for desc in &descriptors {
        // Truncate description for table display.
        let short_desc: String = desc.description.chars().take(40).collect();
        println!("{:<20} {:<20} {}", desc.id, desc.wire_name, short_desc);
    }

    Ok(())
}

fn inspect_tool(tool_id: &str, ctx: &AppContext) -> Result<()> {
    let id = ToolId::new(tool_id);
    let tool = ctx
        .tool_registry
        .get(&id)
        .ok_or_else(|| FrameworkError::config(format!("tool not found: {tool_id}")))?;

    let desc = tool.descriptor();

    println!("Tool: {}", desc.id);
    println!("  Display name: {}", desc.display_name);
    println!("  Wire name:    {}", desc.wire_name);
    println!("  Description:  {}", desc.description);
    println!();

    // Print the JSON schema for the parameters.
    let schema = serde_json::to_string_pretty(&desc.parameters)
        .unwrap_or_else(|_| desc.parameters.to_string());
    println!("Parameters schema:");
    for line in schema.lines() {
        println!("  {line}");
    }

    if !desc.metadata.is_empty() {
        println!();
        println!("Metadata:");
        for (k, v) in &desc.metadata {
            println!("  {k}: {v}");
        }
    }

    Ok(())
}
