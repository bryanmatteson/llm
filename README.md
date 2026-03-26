# llm-framework

Modular Rust toolkit for building LLM-powered applications.

Each crate stands alone. Pull in authentication without sessions, questionnaires without providers, or tool calling without a GUI. Compose them when you need the full stack.

## Features

- **Typed tool system** -- define tools as Rust structs, get JSON Schema generation and validation for free
- **Provider-agnostic sessions** -- turn loop with automatic tool dispatch, approval gating, and timeouts
- **OAuth 2.0 + PKCE and API key auth** -- UI-agnostic auth flows that work in terminals, GUIs, or headless environments
- **Data-driven questionnaires** -- conditional branching, validation rules, pull-based engine with zero framework dependencies
- **Pluggable storage** -- in-memory and file-backed stores for credentials, sessions, and accounts
- **CLI and GUI facades** -- terminal renderers and DTO-oriented async API, both built on the same core

## Architecture

```
                       llm-core
               (IDs, errors, messages)
                         |
       +---------+-------+--------+-----------------+
       |         |                |                  |
   llm-auth  llm-tools     llm-store        llm-provider-api
       |         |            |                      |
       |         |            |                      |
       +---------+-----+------+                      |
                       |                             |
                   llm-session -----------------------+
                       |
              +--------+--------+
              |                 |
   llm-provider-openai      llm-app
   llm-provider-anthropic   /      \
                     llm-cli     llm-gui-api

   llm-questionnaire (standalone, no framework deps)
   llm-config (TOML-based configuration)
```

## Quick Start

```rust,ignore
use std::sync::Arc;
use llm_app::AppBuilder;
use llm_session::SessionBuilder;

// Wire up the application
let ctx = AppBuilder::new()
    .register_provider(openai_registration)
    .register_tool(Arc::new(my_tool))
    .build()?;

// Create a session
let config = SessionBuilder::new("openai")
    .model("gpt-4o")
    .system_prompt("You are a helpful assistant.")
    .build();

let (handle, tx, mut rx) = ctx.sessions
    .create_session(&provider_id, &auth_session, config)
    .await?;
```

---

## Story 1: Authentication & OAuth

**Crate:** `llm-auth` (+ `llm-core` for ID types)

The auth layer never touches the terminal or browser. It returns a descriptor telling you what to show the user, and you call back when the user has acted.

### API Key Auth

```rust,ignore
use llm_auth::{AuthProvider, AuthStart, AuthSession, TokenPair};
use llm_core::Metadata;

// start_login() tells the UI what to do
let start = provider.start_login().await?;

match start {
    AuthStart::ApiKeyPrompt { env_var_hint } => {
        println!("Set {env_var_hint} or enter your key:");
    }
    _ => {}
}

// Pass the key back through complete_login
let mut params = Metadata::new();
params.insert("api_key".into(), "sk-my-key".into());
let completion = provider.complete_login(&params).await?;
let session: AuthSession = completion.session;

// The session holds a TokenPair with expiration tracking
assert!(!session.tokens.is_expired());
assert!(session.tokens.needs_refresh(chrono::Duration::minutes(5)));
```

### OAuth Browser Flow

```rust,ignore
use llm_auth::{AuthProvider, AuthStart};
use llm_core::Metadata;

let start = provider.start_login().await?;

match start {
    AuthStart::OAuthBrowser { url, redirect_uri, state } => {
        // Open url in user's browser, listen for callback on redirect_uri
        println!("Open: {url}");

        // After callback, pass the code and state back
        let mut params = Metadata::new();
        params.insert("code".into(), authorization_code);
        params.insert("state".into(), state);
        let completion = provider.complete_login(&params).await?;

        // TokenPair has access_token, optional refresh_token, and expires_at
        let tokens: &TokenPair = &completion.session.tokens;
        if tokens.can_refresh() {
            // Refresh proactively before expiry
            let refreshed = provider.refresh(&completion.session).await?;
        }
    }
    _ => {}
}
```

### Token Persistence via `llm-store`

```rust,ignore
use llm_store::{CredentialStore, FileCredentialStore, InMemoryCredentialStore};

// In-memory for tests
let store = InMemoryCredentialStore::new();

// File-backed for production
let store = FileCredentialStore::new("/path/to/credentials");
```

---

## Story 2: Sessions & Chat

**Crates:** `llm-session`, `llm-store`

### Build a Session with `SessionBuilder`

```rust,ignore
use llm_session::{SessionBuilder, SessionConfig};
use llm_tools::ToolApproval;

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
    .meta("user", "alice")
    .build();
```

### Create and Manage Sessions

```rust,ignore
use std::sync::Arc;
use llm_session::{DefaultSessionManager, SessionManager, SessionHandle};
use llm_store::InMemorySessionStore;

let store = Arc::new(InMemorySessionStore::new());
let manager = DefaultSessionManager::new(store);

// Create a session -- generates a random UUID and persists to the store
let handle: SessionHandle = manager.create_session(config).await?;
println!("Session: {}", handle.id.as_str());

// Reload later
let restored = manager.get_session(&handle.id).await?;

// List all sessions
let ids = manager.list_sessions().await?;
```

### Run the Turn Loop

```rust,ignore
use llm_session::{run_turn_loop, event_channel, AutoApproveHandler, TurnOutcome};

let (tx, mut rx) = event_channel();
handle.conversation.append_user("What is the weather in NYC?");

let outcome: TurnOutcome = run_turn_loop(
    &handle.id,
    &client,            // impl LlmProviderClient
    &mut handle.conversation,
    &tool_registry,
    &tool_adapter,      // impl ToolSchemaAdapter
    &handle.config,
    &AutoApproveHandler,
    Some(&tx),
).await?;

println!("{}", outcome.final_text);
println!("turns={} tool_calls={}", outcome.turns_used, outcome.tool_calls_made);
println!("tokens: in={} out={}", outcome.usage.input_tokens, outcome.usage.output_tokens);
```

---

## Story 3: Streaming Chat with Tool Calls

**Crate:** `llm-tools` (+ `llm-session` for the turn loop)

### Define a Tool

Input derives `Deserialize` and `JsonSchema` (doc comments become the description the model sees). Output derives `Serialize`.

```rust,ignore
use async_trait::async_trait;
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
        Ok(WeatherOutput {
            temperature: 22.0,
            unit: "celsius".into(),
        })
    }
}
```

### Register Tools

```rust,ignore
use std::sync::Arc;
use llm_tools::{ToolRegistry, DynTool};

let mut registry = ToolRegistry::new();
registry.register(Arc::new(WeatherTool) as Arc<dyn DynTool>);

// Lookup by ID or wire name
let tool = registry.get_by_wire_name("get_weather");

// Get all descriptors (sorted, includes JSON Schema)
let descriptors = registry.all_descriptors();
```

### Configure Tool Policy

```rust,ignore
use llm_tools::{ToolPolicyBuilder, ToolApproval};

let policy = ToolPolicyBuilder::new()
    .default(ToolApproval::RequireConfirmation)
    .allow("weather")
    .allow("search")
    .deny("dangerous_tool")
    .confirm_with("web_search", 10)
    .build();

assert!(policy.is_allowed(&llm_core::ToolId::new("weather")));
assert!(!policy.is_allowed(&llm_core::ToolId::new("dangerous_tool")));
```

### Observe Tool Calls via Events

The mediator emits `SessionEvent`s as tools are called and completed:

```rust,ignore
use llm_session::{event_channel, SessionEvent};

let (tx, mut rx) = event_channel();

// In a separate task:
while let Some(event) = rx.recv().await {
    match event {
        SessionEvent::ToolCallRequested { tool_name, arguments, .. } => {
            eprintln!("[calling {tool_name}]");
        }
        SessionEvent::ToolCallCompleted { tool_name, summary, .. } => {
            eprintln!("[{tool_name} -> {summary}]");
        }
        SessionEvent::ToolApprovalRequired { tool_name, .. } => {
            eprintln!("[{tool_name} needs approval]");
        }
        SessionEvent::TurnCompleted { text, model, usage } => {
            println!("{text}");
            eprintln!("[model={model} tokens={}]", usage.total());
        }
        SessionEvent::TurnLimitReached { turns_used } => {
            eprintln!("[limit reached after {turns_used} turns]");
        }
        SessionEvent::Error { message } => {
            eprintln!("[error: {message}]");
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
use llm_questionnaire::{QuestionnaireBuilder, QuestionnaireRun, AnswerMap, AnswerValue};

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
    .multi_select("features", "Select features:", &["streaming", "logging", "cache"])
    .text("notes", "Anything else?")
    .build();
```

### Run the Engine

The engine is UI-agnostic. You pull the next question and push answers. Conditions are evaluated automatically -- questions whose conditions are not met are skipped.

```rust,ignore
let mut run = QuestionnaireRun::new(questionnaire).unwrap();

while let Some(question) = run.next_question() {
    println!("{}", question.label);

    if let Some(help) = &question.help_text {
        println!("  ({help})");
    }

    // Your UI collects an answer
    let answer = collect_answer_from_ui(question);

    match run.submit_answer(answer) {
        Ok(()) => {}                            // advanced to next question
        Err(errors) => {
            for e in &errors { eprintln!("  {e}"); }
            // question is NOT advanced -- re-ask
        }
    }
}

assert!(run.is_complete());
```

### Read Typed Answers

```rust,ignore
use llm_questionnaire::QuestionId;

let answers: &AnswerMap = run.answers();

let provider: Option<&str>  = answers.choice(&QuestionId::new("provider"));
let api_key: Option<&str>   = answers.text(&QuestionId::new("api_key"));
let tools: Option<bool>     = answers.yes_no(&QuestionId::new("enable_tools"));
let turns: Option<f64>      = answers.number(&QuestionId::new("max_turns"));
```

---

## Story 5: CLI Integration

**Crate:** `llm-cli` (library)

Three reusable building blocks for terminal applications.

### Run a Questionnaire in the Terminal

```rust,ignore
use llm_cli::render::questionnaire::run_terminal_questionnaire;

let answers = run_terminal_questionnaire(&questionnaire)?;
// Renders choices, yes/no prompts, text inputs, and number inputs
// to stderr with defaults, validation, and retry on error.
```

### Render Streaming Session Events

```rust,ignore
use llm_cli::render::stream::render_session_events;
use llm_session::EventReceiver;

let mut rx: EventReceiver = event_receiver;
render_session_events(&mut rx).await;
// Prints streaming deltas inline, tool calls as [calling tool_name],
// results as [tool result: summary], and usage stats on completion.
```

### Terminal Tool Approval

```rust,ignore
use llm_cli::approval::CliApprovalHandler;
use llm_session::ApprovalHandler;

let handler = CliApprovalHandler;
// When a tool call requires confirmation, prints the tool name and
// arguments to stderr and prompts "Allow? [Y/n]" on stdin.
// Pass it as the ApprovalHandler to run_turn_loop.
```

---

## Story 6: GUI Integration

**Crate:** `llm-gui-api`

A DTO-oriented facade for GUI frontends (Tauri, egui, web). Every method returns lightweight, serializable types -- the GUI never depends on internal framework types.

### The Facade

```rust,ignore
use std::sync::Arc;
use llm_gui_api::{GuiFacade, ProviderDto, SessionDto, ToolDto, EventDto, AuthStatusDto};
use llm_app::AppContext;

let facade = GuiFacade::new(Arc::new(app_context));

// Providers dropdown
let providers: Vec<ProviderDto> = facade.list_providers().await?;
// ProviderDto { id, display_name, capabilities }

// Auth status badge
let status: AuthStatusDto = facade.auth_status("openai").await?;
// AuthStatusDto { provider_id, authenticated, method }

// Begin OAuth -- returns JSON the frontend can interpret
let auth_start: serde_json::Value = facade.start_login("openai").await?;

// Complete login
let status = facade.complete_login("openai", params_json).await?;

// Session management
let session: SessionDto = facade.create_session("openai", Some("gpt-4o")).await?;
// SessionDto { id, provider_id, model, message_count }

// Send a message
let event: EventDto = facade.send_message(&session.id, "Hello!").await?;
// EventDto { kind: "turn_completed", data: { "text": "...", ... } }

// Tools panel
let tools: Vec<ToolDto> = facade.list_tools().await?;
// ToolDto { id, display_name, description }
```

### Event Adapter

The `SessionEventAdapter` converts session events into flat `{ kind, data }` DTOs for any transport:

```rust,ignore
use llm_gui_api::SessionEventAdapter;
use llm_gui_api::EventDto;

let event_json = serde_json::to_value(&session_event)?;
let dto: EventDto = SessionEventAdapter::adapt_event(&event_json);
// dto.kind = "tool_call_requested"
// dto.data = { "call_id": "...", "tool_name": "...", "arguments": {...} }
```

---

## Crate Map

| Crate | Purpose | Key Types |
|---|---|---|
| `llm-core` | Shared IDs, errors, message types | `ProviderId`, `SessionId`, `ToolId`, `Message`, `ContentBlock`, `TokenUsage`, `StopReason`, `FrameworkError` |
| `llm-auth` | Provider authentication | `AuthProvider`, `AuthStart`, `AuthSession`, `AuthMethod`, `TokenPair`, `OAuthEndpoints`, `PkceChallenge` |
| `llm-tools` | Typed tool system | `Tool`, `ToolInfo`, `DynTool`, `ToolRegistry`, `ToolPolicy`, `ToolPolicyBuilder`, `ToolApproval`, `ToolContext`, `JsonSchema` |
| `llm-questionnaire` | Data-driven questionnaires | `QuestionnaireBuilder`, `QuestionnaireRun`, `AnswerMap`, `AnswerValue`, `QuestionConfig`, `ConditionExpr` |
| `llm-config` | TOML configuration loading | `AppConfig`, `ProviderConfig`, `SessionDefaults` |
| `llm-store` | Pluggable persistence | `CredentialStore`, `SessionStore`, `AccountStore`, `InMemorySessionStore`, `FileSessionStore` |
| `llm-provider-api` | Provider client trait | `LlmProviderClient`, `TurnRequest`, `TurnResponse`, `ProviderEvent`, `ToolSchemaAdapter` |
| `llm-session` | Turn-loop orchestration | `SessionBuilder`, `SessionConfig`, `SessionHandle`, `SessionManager`, `run_turn_loop`, `run_streaming_turn_loop`, `TurnOutcome`, `SessionEvent`, `ApprovalHandler` |
| `llm-provider-openai` | OpenAI provider | `OpenAiClient`, `OpenAiAuthProvider`, `OpenAiToolFormat` |
| `llm-provider-anthropic` | Anthropic/Claude provider | `AnthropicClient`, `AnthropicAuthProvider`, `AnthropicToolFormat` |
| `llm-provider-google` | Google/Gemini provider | `GoogleClient`, `GoogleAuthProvider`, `GoogleToolFormat` |
| `llm-app` | Application wiring | `AppBuilder`, `AppContext`, `ProviderRegistration`, `ProviderClientFactory`, `AuthService`, `SessionService` |
| `llm-cli` | Terminal utilities | `CliApprovalHandler`, `run_terminal_questionnaire`, `render_session_events` |
| `llm-gui-api` | GUI facade + DTOs | `GuiFacade`, `ProviderDto`, `SessionDto`, `ToolDto`, `EventDto`, `SessionEventAdapter` |

## Adding a New Provider

Implement three traits and bundle them into a `ProviderRegistration`:

```rust,ignore
use llm_auth::AuthProvider;
use llm_provider_api::{LlmProviderClient, ToolSchemaAdapter};
use llm_app::{ProviderClientFactory, ProviderRegistration};

// 1. AuthProvider -- discover, start_login, complete_login, refresh, validate, logout
struct MyAuthProvider { /* ... */ }
impl AuthProvider for MyAuthProvider { /* ... */ }

// 2. LlmProviderClient -- send_turn, stream_turn, list_models
struct MyClient { /* ... */ }
impl LlmProviderClient for MyClient { /* ... */ }

// 3. ProviderClientFactory -- creates a client from an AuthSession + ModelId
struct MyFactory;
#[async_trait]
impl ProviderClientFactory for MyFactory {
    async fn create_client(
        &self,
        auth: &AuthSession,
        model: &ModelId,
    ) -> Result<Box<dyn LlmProviderClient>> {
        Ok(Box::new(MyClient::new(auth, model)))
    }
}

// 4. ToolSchemaAdapter -- translate tool descriptors to your wire format
struct MyAdapter;
impl ToolSchemaAdapter for MyAdapter { /* ... */ }

// 5. Register
let registration = ProviderRegistration {
    descriptor: ProviderDescriptor { /* ... */ },
    auth_provider: Arc::new(MyAuthProvider),
    client_factory: Arc::new(MyFactory),
    tool_adapter: Arc::new(MyAdapter),
};

let ctx = AppBuilder::new()
    .register_provider(registration)
    .build()?;
```

## Adding a New Tool

Implement the `Tool` trait. The framework handles JSON Schema generation, deserialization, serialization, and error reporting.

```rust,ignore
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use llm_tools::{Tool, ToolInfo, ToolContext, JsonSchema};
use llm_core::Result;

#[derive(Debug, Deserialize, JsonSchema)]
struct CalculateInput {
    /// Mathematical expression to evaluate.
    expression: String,
}

#[derive(Debug, Serialize)]
struct CalculateOutput {
    result: f64,
}

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

// Register it
registry.register(Arc::new(CalculatorTool) as Arc<dyn DynTool>);
```

## License

<!-- TODO: choose a license -->
