# llm

Modular Rust toolkit for building LLM-powered applications.

One dependency gets you everything. Or pull in just what you need — authentication without sessions, questionnaires without providers, tool calling without a GUI. Compose them when you need the full stack.

## Install

```toml
# Everything (all providers, CLI + GUI adapters)
[dependencies]
llm = { git = "https://github.com/bryanmatteson/llm.git" }

# Just OpenAI + CLI
llm = { git = "...", default-features = false, features = ["openai", "cli"] }

# Just core (no providers, no frontends)
llm = { git = "...", default-features = false }
```

Or depend on individual crates directly:

```toml
llm-core          = { git = "https://github.com/bryanmatteson/llm.git" }
llm-auth          = { git = "https://github.com/bryanmatteson/llm.git" }
llm-tools         = { git = "https://github.com/bryanmatteson/llm.git" }
llm-questionnaire = { git = "https://github.com/bryanmatteson/llm.git" }
```

### Features

| Feature | Default | What it adds |
|---|---|---|
| `openai` | yes | `llm::openai` — OpenAI / GPT provider |
| `anthropic` | yes | `llm::anthropic` — Anthropic / Claude provider |
| `google` | yes | `llm::google` — Google / Gemini provider |
| `cli` | yes | `llm::cli` — terminal questionnaire renderer, stream renderer, approval handler |
| `gui` | yes | `llm::gui` — DTO facade for GUI frontends (Tauri, egui, web) |

## Quick Start

```rust,ignore
use llm::{AppBuilder, SessionConfig, SessionBuilder, TurnOutcome, Message};
use llm::openai::{openai_registration};

let ctx = AppBuilder::new()
    .with_data_dir("~/.config/my-app")      // persist auth + sessions to disk
    .register_provider(openai_registration())
    .register_tool(Arc::new(my_tool))
    .build()?;

let config = SessionBuilder::new("openai")
    .model("gpt-4o")
    .system_prompt("You are a helpful assistant.")
    .build();

let (handle, tx, rx) = ctx.sessions
    .create_session(&provider_id, &auth_session, config)
    .await?;
```

## Architecture

```
                       llm-core
              (IDs, errors, messages,
            SessionConfig, ToolPolicy,
                 SessionLimits)
                         |
       +---------+-------+--------+-----------+
       |         |                |            |
   llm-auth  llm-tools     llm-store   llm-provider-api
       |         |            |               |
       +---------+-----+------+               |
                       |                      |
                   llm-session ---------------+
                       |
              +--------+--------+
              |                 |
   llm-provider-openai      llm-app
   llm-provider-anthropic   /      \
   llm-provider-google  llm-cli  llm-gui-api
                            \      /
                             llm        <-- facade crate
   llm-questionnaire (standalone)
   llm-config (TOML configuration)
```

---

## Story 1: Authentication & OAuth

**Crate:** `llm-auth` (+ `llm-core` for ID types)

The auth layer never touches the terminal or browser. It returns a descriptor telling you what to show the user, and you call back when the user has acted.

### API Key Auth

```rust,ignore
use llm::auth::{AuthProvider, AuthStart};
use llm::Metadata;

let start = provider.start_login().await?;

match start {
    AuthStart::ApiKeyPrompt { env_var_hint } => {
        println!("Set {env_var_hint} or enter your key:");
    }
    _ => {}
}

let mut params = Metadata::new();
params.insert("api_key".into(), "sk-my-key".into());
let completion = provider.complete_login(&params).await?;
let session = completion.session;

assert!(!session.tokens.is_expired());
```

### OAuth Browser Flow

```rust,ignore
use llm::auth::{AuthProvider, AuthStart};
use llm::Metadata;

let start = provider.start_login().await?;

match start {
    AuthStart::OAuthBrowser { url, redirect_uri, state } => {
        // Open url in user's browser, listen for callback on redirect_uri
        println!("Open: {url}");

        let mut params = Metadata::new();
        params.insert("code".into(), authorization_code);
        params.insert("state".into(), state);
        let completion = provider.complete_login(&params).await?;

        if completion.session.tokens.can_refresh() {
            let refreshed = provider.refresh(&completion.session).await?;
        }
    }
    _ => {}
}
```

### Persistence

```rust,ignore
use llm::store::{FileCredentialStore, InMemoryCredentialStore};

let store = InMemoryCredentialStore::new();           // tests
let store = FileCredentialStore::new("/path/to/dir"); // production

// Or let AppBuilder handle it:
let ctx = AppBuilder::new()
    .with_data_dir("~/.config/my-app")  // credentials, accounts, sessions
    .build()?;
```

---

## Story 2: Sessions & Chat

**Crates:** `llm-session`, `llm-store`

### Build a Session

```rust,ignore
use llm::{SessionBuilder, SessionConfig, ToolApproval};

let config: SessionConfig = SessionBuilder::new("openai")
    .model("gpt-4o")
    .system_prompt("You are a helpful assistant.")
    .max_turns(20)
    .turn_timeout_secs(180)
    .tool_timeout_secs(60)
    .max_tool_calls_per_turn(12)
    .default_tool_approval(ToolApproval::Auto)
    .deny_tool("exec_shell")
    .confirm_tool("delete_file")
    .confirm_tool_with_limit("web_search", 10)
    .build();

// Or the minimal version:
let config = SessionConfig::for_provider("openai");
```

### Create and Manage Sessions

```rust,ignore
use llm::session::{DefaultSessionManager, SessionManager};
use llm::store::InMemorySessionStore;

let store = Arc::new(InMemorySessionStore::new());
let manager = DefaultSessionManager::new(store);

let handle = manager.create_session(config).await?;

// Reload later -- full config (tool policy, limits, system prompt) is restored
let restored = manager.get_session(&handle.id).await?;

let ids = manager.list_sessions().await?;
```

### Run the Turn Loop

```rust,ignore
use llm::session::{run_turn_loop, event_channel, AutoApproveHandler, TurnLoopContext};

let (tx, mut rx) = event_channel();
handle.conversation.append_user("What is the weather in NYC?");

let outcome = run_turn_loop(TurnLoopContext {
    session_id: &handle.id,
    client: client.as_ref(),
    conversation: &mut handle.conversation,
    tool_registry: &tool_registry,
    tool_adapter: tool_adapter.as_ref(),
    config: &handle.config,
    approval_handler: &AutoApproveHandler,
    event_tx: Some(&tx),
}).await?;

println!("{}", outcome.final_text);
println!("turns={} tools={} tokens={}",
    outcome.turns_used, outcome.tool_calls_made, outcome.usage.total());
```

### Streaming

```rust,ignore
use llm::session::run_streaming_turn_loop;

// Same TurnLoopContext -- emits AssistantDelta events for each text chunk.
// Falls back to send_turn automatically if the provider doesn't support streaming.
let outcome = run_streaming_turn_loop(TurnLoopContext {
    session_id: &handle.id,
    client: client.as_ref(),
    conversation: &mut handle.conversation,
    tool_registry: &tool_registry,
    tool_adapter: tool_adapter.as_ref(),
    config: &handle.config,
    approval_handler: &AutoApproveHandler,
    event_tx: Some(&tx),
}).await?;
```

---

## Story 3: Typed Tool Calls

**Crate:** `llm-tools` (+ `llm-session` for the turn loop)

### Define a Tool

Doc comments on input fields become the description the model sees. JSON Schema is generated automatically.

```rust,ignore
use llm::tools::{Tool, ToolInfo, ToolContext, JsonSchema};
use llm::Result;

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

### Register and Configure

```rust,ignore
use llm::{ToolRegistry, DynTool, ToolPolicyBuilder, ToolApproval};

let mut registry = ToolRegistry::new();
registry.register(Arc::new(WeatherTool) as Arc<dyn DynTool>);

let policy = ToolPolicyBuilder::new()
    .default(ToolApproval::RequireConfirmation)
    .allow("weather")
    .deny("dangerous_tool")
    .confirm_with("web_search", 10)
    .build();
```

### Observe Tool Calls

```rust,ignore
use llm::SessionEvent;

while let Some(event) = rx.recv().await {
    match event {
        SessionEvent::ToolCallRequested { tool_name, arguments, .. } => {
            eprintln!("[calling {tool_name}]");
        }
        SessionEvent::ToolCallCompleted { tool_name, summary, .. } => {
            eprintln!("[{tool_name} -> {summary}]");
        }
        SessionEvent::TurnCompleted { text, usage, .. } => {
            println!("{text}");
            eprintln!("[tokens: {}]", usage.total());
        }
        _ => {}
    }
}
```

---

## Story 4: Data-Driven Questionnaires

**Crate:** `llm-questionnaire` (standalone -- zero framework dependencies)

### Build a Questionnaire

```rust,ignore
use llm::questionnaire::{QuestionnaireBuilder, QuestionnaireRun, AnswerValue};

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
    .number_with("max_turns", "Max turns?", |q| q.range(1.0, 100.0).default_number(10.0))
    .multi_select("features", "Select features:", &["streaming", "logging", "cache"])
    .text("notes", "Anything else?")
    .build();
```

### Run the Engine

The engine is UI-agnostic. Pull the next question, push an answer. Conditions are evaluated automatically.

```rust,ignore
let mut run = QuestionnaireRun::new(questionnaire).unwrap();

while let Some(question) = run.next_question() {
    let answer = collect_answer_from_ui(question);
    match run.submit_answer(answer) {
        Ok(()) => {}
        Err(errors) => { /* validation failed -- re-ask */ }
    }
}

let answers = run.answers();
let provider: Option<&str> = answers.choice(&QuestionId::new("provider"));
let api_key: Option<&str>  = answers.text(&QuestionId::new("api_key"));
let tools: Option<bool>    = answers.yes_no(&QuestionId::new("enable_tools"));
```

---

## Story 5: CLI Integration

**Crate:** `llm-cli` (library)

Three reusable building blocks for terminal applications.

```rust,ignore
use llm::cli::render::questionnaire::run_terminal_questionnaire;
use llm::cli::render::stream::render_session_events;
use llm::cli::approval::CliApprovalHandler;

// Run a questionnaire interactively in the terminal
let answers = run_terminal_questionnaire(&questionnaire)?;

// Render streaming session events (deltas, tool calls, completion)
render_session_events(&mut rx).await;

// Prompt "Allow? [Y/n]" for tool calls requiring confirmation
let handler = CliApprovalHandler;
```

---

## Story 6: GUI Integration

**Crate:** `llm-gui-api`

A DTO-oriented facade for GUI frontends. Every method returns lightweight, serializable types.

```rust,ignore
use llm::gui::{GuiFacade, ProviderDto, SessionDto, EventDto};

let facade = GuiFacade::new(Arc::new(app_context));

let providers: Vec<ProviderDto> = facade.list_providers().await?;
let status = facade.auth_status("openai").await?;
let session: SessionDto = facade.create_session("openai", Some("gpt-4o")).await?;
let event: EventDto = facade.send_message(&session.id, "Hello!").await?;
```

---

## Crate Map

| Crate | Purpose | Key Types |
|---|---|---|
| `llm` | **Facade** — single dependency, feature-gated re-exports | Everything below |
| `llm-core` | Shared IDs, errors, messages, config, policy | `ProviderId`, `SessionId`, `ToolId`, `Message`, `SessionConfig`, `ToolPolicy`, `SessionLimits`, `FrameworkError` |
| `llm-auth` | Provider authentication | `AuthProvider`, `AuthStart`, `AuthSession`, `TokenPair`, `PkceChallenge` |
| `llm-tools` | Typed tool system | `Tool`, `ToolInfo`, `DynTool`, `ToolRegistry`, `ToolPolicyBuilder`, `ToolApproval` |
| `llm-questionnaire` | Data-driven questionnaires | `QuestionnaireBuilder`, `QuestionnaireRun`, `AnswerMap`, `AnswerValue`, `ConditionExpr` |
| `llm-config` | TOML configuration | `AppConfig`, `ProviderConfig`, `SessionDefaults`, `ConfigLoader` |
| `llm-store` | Pluggable persistence | `CredentialStore`, `SessionStore`, `AccountStore`, `FileCredentialStore`, `InMemorySessionStore` |
| `llm-provider-api` | Provider client trait | `LlmProviderClient`, `TurnRequest`, `TurnResponse`, `ProviderEvent`, `ToolSchemaAdapter` |
| `llm-session` | Turn-loop orchestration | `SessionBuilder`, `TurnLoopContext`, `run_turn_loop`, `run_streaming_turn_loop`, `TurnOutcome`, `SessionEvent` |
| `llm-provider-openai` | OpenAI provider | `OpenAiClient`, `OpenAiAuthProvider`, `OpenAiToolFormat` |
| `llm-provider-anthropic` | Anthropic/Claude provider | `AnthropicClient`, `AnthropicAuthProvider`, `AnthropicToolFormat` |
| `llm-provider-google` | Google/Gemini provider | `GoogleClient`, `GoogleAuthProvider`, `GoogleToolFormat` |
| `llm-app` | Application wiring | `AppBuilder`, `AppContext`, `ProviderRegistration`, `AuthService`, `SessionService` |
| `llm-cli` | Terminal utilities | `CliApprovalHandler`, `run_terminal_questionnaire`, `render_session_events` |
| `llm-gui-api` | GUI facade + DTOs | `GuiFacade`, `ProviderDto`, `SessionDto`, `EventDto`, `SessionEventAdapter` |

## Adding a New Provider

Implement three traits and bundle them into a `ProviderRegistration`:

```rust,ignore
use llm::auth::AuthProvider;
use llm::provider_api::{LlmProviderClient, ToolSchemaAdapter};
use llm::app::{ProviderClientFactory, ProviderRegistration};

// 1. AuthProvider -- discover, start_login, complete_login, refresh, validate, logout
// 2. LlmProviderClient -- send_turn, stream_turn, list_models
// 3. ProviderClientFactory -- creates a client from AuthSession + ModelId
// 4. ToolSchemaAdapter -- translate tool descriptors to your wire format

let registration = ProviderRegistration {
    descriptor: ProviderDescriptor { /* ... */ },
    auth_provider: Arc::new(MyAuthProvider),
    client_factory: Arc::new(MyFactory),
    tool_adapter: Arc::new(MyAdapter),
};

let ctx = AppBuilder::new()
    .with_data_dir("~/.config/my-app")
    .register_provider(registration)
    .build()?;
```

## Adding a New Tool

```rust,ignore
#[derive(Debug, Deserialize, JsonSchema)]
struct CalculateInput {
    /// Mathematical expression to evaluate.
    expression: String,
}

#[derive(Debug, Serialize)]
struct CalculateOutput { result: f64 }

#[derive(Debug)]
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    type Input = CalculateInput;
    type Output = CalculateOutput;

    fn info(&self) -> ToolInfo {
        ToolInfo::new("calculator", "Evaluate a math expression")
    }

    async fn execute(&self, input: CalculateInput, _ctx: &ToolContext) -> Result<CalculateOutput> {
        let result = evaluate(&input.expression)?;
        Ok(CalculateOutput { result })
    }
}

registry.register(Arc::new(CalculatorTool) as Arc<dyn DynTool>);
```
