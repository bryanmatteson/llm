use std::io::{self, BufRead, Write};

use clap::{Args, Subcommand};

use llm_core::{FrameworkError, Metadata, Result};

use crate::bootstrap::AppContext;

#[derive(Args)]
pub struct AuthArgs {
    #[command(subcommand)]
    pub command: AuthCommands,
}

#[derive(Subcommand)]
pub enum AuthCommands {
    /// Start a login flow for a provider.
    Login {
        /// Provider identifier (e.g. "openai").
        provider: String,

        /// If set, print the URL instead of opening a browser.
        #[arg(long, default_value_t = false)]
        no_browser: bool,
    },

    /// Log out from a provider.
    Logout {
        /// Provider identifier.
        provider: String,
    },

    /// List known accounts.
    List,

    /// Show credential status for all registered providers.
    Status,
}

pub async fn run_auth(args: AuthArgs, ctx: &AppContext) -> Result<()> {
    match args.command {
        AuthCommands::Login {
            provider,
            no_browser,
        } => login(&provider, no_browser, ctx).await,
        AuthCommands::Logout { provider } => logout(&provider, ctx).await,
        AuthCommands::List => list_accounts(ctx).await,
        AuthCommands::Status => status(ctx).await,
    }
}

// ---------------------------------------------------------------------------
// Login
// ---------------------------------------------------------------------------

async fn login(provider: &str, no_browser: bool, ctx: &AppContext) -> Result<()> {
    let auth = ctx
        .auth_provider(provider)
        .ok_or_else(|| FrameworkError::auth(format!("unknown provider: {provider}")))?;

    let start = auth.start_login().await?;

    match start {
        llm_auth::AuthStart::OAuthBrowser {
            url,
            redirect_uri: _,
            state,
        } => {
            if no_browser {
                eprintln!("Open this URL in your browser to authenticate:");
                eprintln!("  {url}");
            } else {
                eprintln!("Opening browser for authentication...");
                eprintln!("  {url}");
                // Best-effort: try to open the URL.  If this fails the user
                // can copy the URL printed above.
                let _ = open_url(&url);
            }

            eprintln!();
            eprintln!("After authenticating, paste the callback code below.");
            eprint!("Code: ");
            io::stderr().flush().ok();

            let mut code = String::new();
            io::stdin()
                .lock()
                .read_line(&mut code)
                .map_err(|e| FrameworkError::auth(format!("failed to read code: {e}")))?;
            let code = code.trim().to_owned();

            let mut params = Metadata::new();
            params.insert("code".into(), code);
            params.insert("state".into(), state);

            let completion = auth.complete_login(&params).await?;

            // Persist the session in the credential store.
            ctx.credential_store
                .set_auth_session(auth.provider_id(), &completion.session)
                .await?;

            eprintln!("Logged in to {provider} successfully.");
        }
        llm_auth::AuthStart::ApiKeyPrompt { env_var_hint } => {
            eprintln!("Enter your API key (hint: set {env_var_hint} to skip this prompt):");
            eprint!("API key: ");
            io::stderr().flush().ok();

            let mut key = String::new();
            io::stdin()
                .lock()
                .read_line(&mut key)
                .map_err(|e| FrameworkError::auth(format!("failed to read API key: {e}")))?;
            let key = key.trim().to_owned();

            let mut params = Metadata::new();
            params.insert("api_key".into(), key);

            let completion = auth.complete_login(&params).await?;
            ctx.credential_store
                .set_auth_session(auth.provider_id(), &completion.session)
                .await?;

            eprintln!("API key saved for {provider}.");
        }
        llm_auth::AuthStart::DeviceCode {
            verification_uri,
            user_code,
            interval: _,
        } => {
            eprintln!("Go to: {verification_uri}");
            eprintln!("Enter code: {user_code}");
            eprintln!("Waiting for authorisation (press Enter once complete)...");

            let mut buf = String::new();
            io::stdin().lock().read_line(&mut buf).ok();

            let params = Metadata::new();
            let completion = auth.complete_login(&params).await?;
            ctx.credential_store
                .set_auth_session(auth.provider_id(), &completion.session)
                .await?;

            eprintln!("Logged in to {provider} successfully.");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Logout
// ---------------------------------------------------------------------------

async fn logout(provider: &str, ctx: &AppContext) -> Result<()> {
    let auth = ctx
        .auth_provider(provider)
        .ok_or_else(|| FrameworkError::auth(format!("unknown provider: {provider}")))?;

    // If we have an active session, let the provider clean up.
    if let Some(session) = ctx
        .credential_store
        .get_auth_session(auth.provider_id())
        .await?
    {
        auth.logout(&session).await?;
    }

    ctx.credential_store
        .clear_auth_session(auth.provider_id())
        .await?;
    ctx.credential_store
        .clear_api_key(auth.provider_id())
        .await?;

    eprintln!("Logged out from {provider}.");
    Ok(())
}

// ---------------------------------------------------------------------------
// List accounts
// ---------------------------------------------------------------------------

async fn list_accounts(ctx: &AppContext) -> Result<()> {
    let accounts = ctx.account_store.list_accounts().await?;

    if accounts.is_empty() {
        println!("No accounts registered.");
        return Ok(());
    }

    println!("{:<16} {:<24} CREATED", "PROVIDER", "NAME");
    println!("{}", "-".repeat(60));
    for acct in &accounts {
        println!(
            "{:<16} {:<24} {}",
            acct.provider_id,
            acct.display_name,
            acct.created_at.format("%Y-%m-%d %H:%M"),
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

async fn status(ctx: &AppContext) -> Result<()> {
    println!("{:<16} {:<12} AUTH SESSION", "PROVIDER", "API KEY");
    println!("{}", "-".repeat(44));

    for desc in &ctx.provider_descriptors {
        let status = ctx.credential_store.credential_status(&desc.id).await?;

        let key_marker = if status.has_api_key { "yes" } else { "no" };
        let session_marker = if status.has_auth_session { "yes" } else { "no" };

        println!("{:<16} {:<12} {}", desc.id, key_marker, session_marker);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Best-effort URL opener.
fn open_url(url: &str) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    return std::process::Command::new("open")
        .arg(url)
        .spawn()
        .map(|_| ());
    #[cfg(target_os = "linux")]
    return std::process::Command::new("xdg-open")
        .arg(url)
        .spawn()
        .map(|_| ());
    #[cfg(target_os = "windows")]
    return std::process::Command::new("cmd")
        .args(["/C", "start", url])
        .spawn()
        .map(|_| ());
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    return Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "cannot open URLs on this platform",
    ));
}
