# llm-integration

A modular Rust framework for LLM-powered applications. Authenticate with providers, run streaming conversations with tool calls, collect structured input through questionnaires, and integrate it all into CLI or GUI frontends through shared, UI-agnostic core logic.

## Flows

The framework is organized around six concrete flows. Each section shows what it does, what crates are involved, and how to use it.

---

### 1. Auth and OAuth

Authenticate with LLM providers via OAuth 2.0 (with PKCE) or API keys. The auth layer never touches the terminal or the browser — it returns a descriptor telling the UI what to do, and the UI calls back when the user has acted.

**Crates:** `llm-auth`, `llm-core`

```rust
use llm_auth::{AuthProvider, AuthStart};

// 1. Start login — get instructions for the UI
let start = provider.start_login().await?;

match start {
    AuthStart::OAuthBrowser { url, state, .. } => {
        // Open the URL in a browser, wait for callback
        println!("Open: {url}");
    }
    AuthStart::ApiKeyPrompt { env_var_hint } => {
        // Prompt the user for an API key
        println!("Set {env_var_hint} or enter your key:");
    }
    AuthStart::DeviceCode { user_code, verification_uri, .. } => {
        // Show device code
        println!("Enter {user_code} at {verification_uri}");
    }
}

// 2. Complete login — pass the user's response back
let completion = provider.complete_login(&params).await?;
let session = completion.session; // AuthSession with tokens

// 3. Later: validate and refresh
if !provider.validate(&session).await? {
    let refreshed = provider.refresh(&session).await?;
}
```

OAuth primitives (PKCE challenges, endpoint configs, token exchange) are reusable independently. PKCE and state generation use OS-level CSPRNG via `getrandom`.

---

### 2. Sessions

Create configured interaction contexts with a provider, model, system prompt, and tool policy. Sessions own the conversation transcript and are persistable through pluggable stores.

**Crates:** `llm-session`, `llm-store`, `llm-core`

```rust
use llm_session::{SessionBuilder, DefaultSessionManager};
use llm_tools::ToolApproval;
use llm_store::InMemorySessionStore;

let config = SessionBuilder::new("openai")
    .model("gpt-4o-mini")
    .system_prompt("You are a helpful assistant.")
    .max_turns(20)
    .default_tool_approval(ToolApproval::Auto)
    .confirm_tool_with_limit("delete_file", 5)
    .deny_tool("exec_shell")
    .build();

let manager = DefaultSessionManager::new(Arc::new(InMemorySessionStore::new()));
let handle = manager.create_session(config).await?;
```

Sessions persist to the store on creation. The `SessionSnapshot` captures the transcript and can be reloaded across process restarts.

The builder handles model selection, system prompts, turn limits, timeouts, and per-tool policy rules. You can also construct `SessionConfig` directly as a struct literal for full control.

---

### 3. Streaming chat with tool calls

The turn loop is the core orchestration function. It sends the conversation to the provider, dispatches tool calls through the registry subject to policy and approval, feeds results back, and repeats until the model produces a final response or a limit is reached.

**Crates:** `llm-session`, `llm-provider-api`, `llm-tools`

```rust
use llm_session::{run_turn_loop, event_channel, AutoApproveHandler, SessionEvent};

// Set up
let (tx, mut rx) = event_channel();
let approval = AutoApproveHandler;

// Spawn event consumer
tokio::spawn(async move {
    while let Some(event) = rx.recv().await {
        match event {
            SessionEvent::AssistantDelta { text } => print!("{text}"),
            SessionEvent::ToolCallRequested { tool_name, .. } => {
                eprintln!("[calling {tool_name}]");
            }
            SessionEvent::ToolApprovalRequired { tool_name, .. } => {
                eprintln!("[{tool_name} requires approval]");
            }
            SessionEvent::TurnCompleted { usage, .. } => {
                eprintln!("[tokens: {}]", usage.total());
            }
            _ => {}
        }
    }
});

// Run the loop
let outcome = run_turn_loop(
    &handle.id,
    &client,              // impl LlmProviderClient
    &mut conversation,
    &tool_registry,
    &tool_adapter,        // impl ToolSchemaAdapter
    &config,
    &approval,            // impl ApprovalHandler
    Some(&tx),
).await?;

println!("Final: {}", outcome.final_text);
println!("Turns: {}, Tool calls: {}", outcome.turns_used, outcome.tool_calls_made);
```

The mediator enforces `turn_timeout` and `tool_timeout` from `SessionLimits`. Skipped tool calls (when the per-turn cap is hit) get error results sent back to the model so the conversation stays well-formed.

Tools are typed. You define Rust structs for input and output, and the framework handles JSON Schema generation, deserialization, and serialization automatically:

```rust
use serde::{Deserialize, Serialize};
use llm_tools::{Tool, ToolInfo, ToolContext, JsonSchema};
use llm_core::Result;

#[derive(Debug, Deserialize, JsonSchema)]
struct WeatherInput {
    /// The city to look up weather for.
    city: String,
}

#[derive(Debug, Serialize)]
struct WeatherOutput {
    temperature: f64,
    unit: String,
}

#[derive(Debug)]
struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    type Input = WeatherInput;
    type Output = WeatherOutput;

    fn info(&self) -> ToolInfo {
        ToolInfo::new("weather", "Get current weather for a city")
            .wire_name("get_weather")
            .display_name("Weather Lookup")
    }

    async fn execute(&self, input: WeatherInput, _ctx: &ToolContext) -> Result<WeatherOutput> {
        Ok(WeatherOutput { temperature: 22.0, unit: "celsius".into() })
    }
}
```

The JSON Schema sent to the LLM is derived from `#[derive(JsonSchema)]` — doc comments on fields become the `description` the model sees. Invalid input from the model produces clear deserialization errors returned as tool results, not panics.

---

### 4. Questionnaires

A standalone, data-driven engine for collecting structured user input. Supports choice, yes/no, text, number, and multi-select questions with conditional branching, validation rules, and a pull-based API that any UI can drive.

**Crate:** `llm-questionnaire` (zero framework dependencies — usable in any Rust project)

```rust
use llm_questionnaire::*;

let questionnaire = QuestionnaireBuilder::new("setup", "Provider Setup")
    .description("Configure your LLM provider")
    .choice_with("provider", "Which provider?", &["openai", "anthropic"], |q| {
        q.default("openai").required()
    })
    .text_with("api_key", "API key:", |q| {
        q.placeholder("sk-...")
            .min_length(8)
            .show_if_equals("provider", "openai")
            .required()
    })
    .yes_no_with("enable_tools", "Enable tool calling?", |q| q.default_yes())
    .number_with("max_turns", "Max conversation turns?", |q| {
        q.range(1.0, 100.0).default_number(10.0)
    })
    .text("notes", "Anything else?")
    .build();

// The engine is UI-agnostic — you drive it with next_question/submit_answer
let mut run = QuestionnaireRun::new(questionnaire).unwrap();

while let Some(question) = run.next_question() {
    // Your UI presents the question and collects input
    let answer = get_answer_from_user(question);

    match run.submit_answer(answer) {
        Ok(()) => {}                              // advanced to next
        Err(errors) => show_errors(errors),       // re-ask same question
    }
}

let answers: &AnswerMap = run.answers();
```

The builder handles option lists, defaults, validation rules, and conditional visibility in a fluent chain. Questionnaires are also serializable to JSON/TOML, so they can be defined in config files and loaded at runtime. You can construct `Questionnaire` as a struct literal for full control.

---

### 5. CLI facade

`llm-cli` is both a binary (`llmctl`) and a library. The library exposes terminal-specific utilities that any CLI application can reuse:

**Crate:** `llm-cli` (library)

```rust
use llm_cli::render::questionnaire::run_terminal_questionnaire;
use llm_cli::render::stream::render_session_events;
use llm_cli::approval::CliApprovalHandler;

// Drive a questionnaire interactively in the terminal
let answers = run_terminal_questionnaire(&my_questionnaire)?;

// Render session events (streaming deltas, tool calls, completions) to stderr
let mut rx = event_receiver;
render_session_events(&mut rx).await;

// Prompt "Allow? [Y/n]" when a tool call needs confirmation
let approval_handler = CliApprovalHandler;
```

The `llmctl` binary uses these same building blocks:

```sh
llmctl auth login --provider openai     # OAuth browser flow or API key prompt
llmctl auth status                      # Show credential status
llmctl session new --provider openai    # Interactive chat with tool calls
llmctl questionnaire run setup          # Run the setup questionnaire
llmctl tools list                       # Show registered tools
llmctl debug providers                  # Show providers and capabilities
```

---

### 6. GUI facade

`llm-gui-api` wraps the application services behind a DTO-oriented async API. Every method returns lightweight, serializable types — so a GUI frontend (Tauri, egui, web) never depends on internal framework types.

**Crate:** `llm-gui-api`

```rust
use llm_gui_api::{GuiFacade, ProviderDto, SessionDto, EventDto};

let facade = GuiFacade::new(Arc::new(app_context));

// List providers for a dropdown
let providers: Vec<ProviderDto> = facade.list_providers().await?;

// Check auth state for a status badge
let status = facade.auth_status("openai").await?;

// Start OAuth — returns JSON the frontend can act on
let auth_start = facade.start_login("openai").await?;

// Create a session
let session: SessionDto = facade.create_session("openai", Some("gpt-4o")).await?;

// Send a message — returns an event DTO with the outcome
let event: EventDto = facade.send_message(&session.id, "Hello").await?;

// List tools for a settings panel
let tools: Vec<ToolDto> = facade.list_tools().await?;
```

The CLI facade and GUI facade are symmetric — both consume `llm-app` services, one translates to terminal I/O, the other translates to DTOs.

---

## Getting started

```sh
# Build everything
cargo build --workspace

# Run tests
cargo test --workspace

# Try the CLI
cargo run -p llm-cli -- debug providers
cargo run -p llm-cli -- auth login --provider openai
cargo run -p llm-cli -- session new --provider openai
```

## Architecture

```
  llm-questionnaire              llm-core
  (standalone)            (IDs, errors, messages)
                                 |
         ┌──────────┬────────────┼──────────┬──────────┐
     llm-auth   llm-tools    llm-config  llm-provider-api
         |          |                           |
         └──────┐   |                           |
            llm-store                           |
                |                               |
            llm-session ────────────────────────┘
                |                     |
        llm-provider-openai       llm-app
                                 /       \
                         llm-cli(lib)   llm-gui-api
                            |
                        llmctl(bin)
```

### Use just what you need

| What you want | What you add | Framework deps |
|---|---|---|
| Questionnaires | `llm-questionnaire` | none |
| Auth / OAuth | `llm-auth` + `llm-core` | minimal |
| Tool registry + policy | `llm-tools` + `llm-core` | minimal |
| Streaming chat with tools | `llm-session` + provider + `llm-tools` | moderate |
| Full app with services | `llm-app` | everything |
| CLI terminal utilities | `llm-cli` (library) | full |
| GUI integration layer | `llm-gui-api` | full |

## Project structure

```
crates/
  llm-core/             IDs, errors, messages, provider/model descriptors
  llm-auth/             AuthProvider trait, OAuth (PKCE), API key resolution, tokens
  llm-tools/            Tool trait, registry, policy, invocation, validation
  llm-questionnaire/    Questionnaire schema, conditions, engine, validation (standalone)
  llm-config/           AppConfig, ProviderConfig, SessionDefaults, TOML loader
  llm-provider-api/     LlmProviderClient trait, TurnRequest/Response, streaming events
  llm-store/            CredentialStore, SessionStore, in-memory + file-backed impls
  llm-session/          Turn-loop mediator, approval, timeouts, events
  llm-provider-openai/  OpenAI auth, chat completions client, tool schema adapter
  llm-app/              ProviderRegistry, AuthService, SessionService, AppBuilder
  llm-cli/              Library: terminal renderers, approval handler, bootstrap
                        Binary: llmctl with auth/session/tools/questionnaire/debug commands
  llm-gui-api/          GuiFacade, DTOs, event adapter for GUI frontends
```
