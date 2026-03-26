# llm-integration

A modular Rust framework for building applications that authenticate with LLM providers, manage interactive sessions, call tools, and collect structured user input through data-driven questionnaires.

The framework separates concerns into focused crates with clean dependency boundaries, making it possible to share the same domain logic across CLI, GUI, and server frontends without duplicating business rules.

## What it does

- **Authenticate** with LLM providers via OAuth 2.0 (with PKCE) or API keys
- **Persist credentials** securely through pluggable storage backends
- **Create sessions** with configurable models, system prompts, and tool policies
- **Execute multi-turn conversations** with automatic tool-call mediation
- **Enforce policies** on which tools are allowed, which require confirmation, and which are denied
- **Collect structured input** through a data-driven questionnaire engine with conditional branching
- **Stream events** from the turn loop to any UI layer (CLI, GUI, or tests)

## Architecture

```
  llm-questionnaire             llm-core
  (standalone)           (IDs, errors, messages)
                                |
         ┌──────────┬───────────┼──────────┬──────────┐
     llm-auth   llm-tools   llm-config  llm-provider-api
         |          |                          |
         └──────┐   |                          |
            llm-store                          |
                |                              |
            llm-session ───────────────────────┘
                |                    |
        llm-provider-openai      llm-app
                                /        \
                          llm-cli      llm-gui-api
```

**12 crates**, each with a single responsibility:

| Crate | Purpose | Standalone? |
|-------|---------|-------------|
| `llm-core` | Shared types: strongly-typed IDs, `FrameworkError`, `Message`, `ProviderDescriptor`, capabilities | --- |
| `llm-auth` | Auth framework: `AuthProvider` trait, OAuth primitives (PKCE, endpoints, flows), API key resolution | + `llm-core` |
| `llm-tools` | Tool system: `Tool` trait, `ToolRegistry`, `ToolPolicy`, invocation, input validation | + `llm-core` |
| `llm-questionnaire` | Data-driven questionnaires: schema, conditional branching, validation, pull-based engine | **fully independent** |
| `llm-config` | Typed configuration: `AppConfig`, `ProviderConfig`, `SessionDefaults`, TOML loader | + `llm-core` |
| `llm-provider-api` | Provider contracts: `LlmProviderClient` trait, `TurnRequest`/`TurnResponse`, streaming events, `ToolSchemaAdapter` | + `llm-core` |
| `llm-store` | Storage abstractions: `CredentialStore`, `AccountStore`, `SessionStore` with in-memory and file-backed implementations | + `llm-core`, `llm-auth` |
| `llm-session` | Session orchestration: turn-loop mediator, approval protocol, timeout enforcement, event streaming | + core, auth, tools, store, provider-api |
| `llm-provider-openai` | OpenAI provider: auth, chat completions client, model discovery, tool schema translation | + core, auth, tools, provider-api |
| `llm-app` | Application services: `ProviderRegistry`, `AuthService`, `SessionService`, `AppBuilder` | all domain crates |
| `llm-cli` | CLI binary (`llmctl`): terminal questionnaire renderer, OAuth callback handler, interactive chat | + `llm-app` |
| `llm-gui-api` | GUI facade: async DTOs and event adapters for GUI frontends | + `llm-app` |

### Use just what you need

**Questionnaire engine only** --- `llm-questionnaire` has zero framework dependencies. Add it to any Rust project for data-driven forms with conditional branching, validation, and a pull-based engine. Its only dependencies are `serde`, `serde_json`, and `regex-lite`.

```toml
[dependencies]
llm-questionnaire = { path = "crates/llm-questionnaire" }
```

**Auth only** --- `llm-auth` depends only on `llm-core` (lightweight ID + error types). Use it for OAuth 2.0 with PKCE, API key resolution, and token management in any application.

```toml
[dependencies]
llm-auth = { path = "crates/llm-auth" }
llm-core = { path = "crates/llm-core" }
```

**Tools only** --- `llm-tools` depends only on `llm-core`. Use it for a typed tool registry with JSON Schema descriptors, policy enforcement, and input validation.

```toml
[dependencies]
llm-tools = { path = "crates/llm-tools" }
llm-core = { path = "crates/llm-core" }
```

**Streaming chat with tool calls** --- Combine `llm-session` + a provider crate + `llm-tools` for the full turn loop with tool mediation, approval, timeouts, and event streaming.

### Key design constraints

- **Provider crates never depend on `llm-session`** --- they implement contracts from `llm-provider-api`
- **`llm-session` is the sole owner of the turn loop** --- all tool mediation, approval, and timeout enforcement lives here
- **`ApprovalHandler` is interaction only, never policy** --- `ToolPolicy` decides; the handler executes the "confirm" case
- **Questionnaire answers are transient** --- only the mapped results (e.g. `SessionConfig`) are persisted
- **PKCE and OAuth state use OS-level CSPRNG** (`getrandom`) --- not a userspace PRNG
- **`llm-questionnaire` is fully framework-independent** --- zero coupling to LLM types, usable in any Rust project

## Getting started

### Prerequisites

- Rust 1.85+ (edition 2024)
- An OpenAI API key or OAuth credentials

### Build the workspace

```sh
cargo build --workspace
```

### Run the CLI

```sh
# See available commands
cargo run -p llm-cli -- --help

# List registered providers and their capabilities
cargo run -p llm-cli -- debug providers

# Check auth status
cargo run -p llm-cli -- auth status

# Log in to OpenAI with an API key
cargo run -p llm-cli -- auth login --provider openai

# Start an interactive chat session
cargo run -p llm-cli -- session new --provider openai

# Run the setup questionnaire
cargo run -p llm-cli -- questionnaire run setup
```

### Run the tests

```sh
cargo test --workspace
```

## Usage

### Wiring up an application

Use `AppBuilder` to compose stores, providers, and tools into a running application context:

```rust
use llm_app::{AppBuilder, ProviderRegistration};
use llm_provider_openai::{OpenAiAuthProvider, OpenAiToolFormat, provider_descriptor};

let ctx = AppBuilder::new()
    .register_provider(ProviderRegistration {
        descriptor: provider_descriptor(),
        auth_provider: Arc::new(OpenAiAuthProvider::new()),
        client_factory: Arc::new(my_client_factory),
        tool_adapter: Arc::new(OpenAiToolFormat),
    })
    .register_tool(Arc::new(my_tool))
    .build()?;

// Use the services
let accounts = ctx.auth.list_accounts().await?;
let session = ctx.sessions.create_session(provider_id, auth, config).await?;
```

### Implementing a tool

Tools implement a simple async trait with a JSON Schema descriptor:

```rust
use async_trait::async_trait;
use llm_tools::{Tool, ToolDescriptor, ToolContext};
use llm_core::{Result, ToolId, Metadata};
use serde_json::{json, Value};

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            id: ToolId::new("weather"),
            wire_name: "get_weather".into(),
            display_name: "Weather Lookup".into(),
            description: "Get current weather for a city".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
            metadata: Metadata::new(),
        }
    }

    async fn execute(
        &self,
        args: Value,
        _ctx: &ToolContext,
    ) -> Result<Value> {
        let city = args["city"].as_str().unwrap_or("unknown");
        Ok(json!({ "temperature": 22, "city": city, "unit": "celsius" }))
    }
}
```

### Defining a questionnaire

Questionnaires are data-driven schemas with conditional branching:

```rust
use llm_questionnaire::*;
use llm_core::{QuestionnaireId, QuestionId};

let questionnaire = Questionnaire {
    id: QuestionnaireId::new("setup"),
    title: "Provider Setup".into(),
    description: "Configure your LLM provider".into(),
    questions: vec![
        Question {
            id: QuestionId::new("provider"),
            label: "Which provider?".into(),
            help_text: None,
            kind: QuestionKind::Choice {
                options: vec![
                    ChoiceOption { value: "openai".into(), label: "OpenAI".into() },
                    ChoiceOption { value: "anthropic".into(), label: "Anthropic".into() },
                ],
                default: Some("openai".into()),
            },
            required: true,
            validation: vec![],
            condition: None,
        },
        Question {
            id: QuestionId::new("api_key"),
            label: "Enter your API key:".into(),
            help_text: Some("Starts with sk-".into()),
            kind: QuestionKind::Text { placeholder: Some("sk-...".into()) },
            required: true,
            validation: vec![ValidationRule::MinLength(8)],
            // Only shown when provider is "openai"
            condition: Some(ConditionExpr::Equals {
                question_id: QuestionId::new("provider"),
                value: serde_json::json!("openai"),
            }),
        },
    ],
};

// Run it (UI-agnostic engine)
let mut run = QuestionnaireRun::new(questionnaire).unwrap();
while let Some(question) = run.next_question() {
    // Present question to user, get answer...
    run.submit_answer(AnswerValue::Choice("openai".into())).unwrap();
}
let answers = run.answers();
```

### Configuring tool policy

Control which tools are allowed, which need confirmation, and which are denied:

```rust
use llm_tools::{ToolPolicy, ToolPolicyRule, ToolApproval};
use llm_core::ToolId;

let policy = ToolPolicy {
    default_approval: ToolApproval::Auto,
    rules: vec![
        ToolPolicyRule {
            tool_id: ToolId::new("delete_file"),
            approval: ToolApproval::RequireConfirmation,
            max_calls_per_session: Some(5),
        },
        ToolPolicyRule {
            tool_id: ToolId::new("exec_shell"),
            approval: ToolApproval::Deny,
            max_calls_per_session: None,
        },
    ],
};
```

### Implementing a provider

Provider crates implement traits from `llm-provider-api` and `llm-auth`:

```rust
use async_trait::async_trait;
use llm_provider_api::{LlmProviderClient, TurnRequest, TurnResponse};
use llm_core::{ProviderId, ModelDescriptor, Result};

struct MyProviderClient { /* ... */ }

#[async_trait]
impl LlmProviderClient for MyProviderClient {
    fn provider_id(&self) -> &ProviderId { /* ... */ }

    async fn send_turn(&self, request: &TurnRequest) -> Result<TurnResponse> {
        // Translate request to provider wire format
        // POST to provider API
        // Normalize response into canonical Message types
    }

    async fn stream_turn(&self, request: &TurnRequest)
        -> Result<Pin<Box<dyn Stream<Item = Result<ProviderEvent>> + Send>>>
    {
        // SSE streaming variant
    }

    async fn list_models(&self) -> Result<Vec<ModelDescriptor>> {
        // GET /models
    }
}
```

### Listening to session events

The turn loop emits events through a channel so any UI can observe progress:

```rust
use llm_session::{SessionEvent, event_channel};

let (tx, mut rx) = event_channel();

// In a separate task:
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
```

## Project structure

```
llm-integration/
  Cargo.toml              # Workspace root
  crates/
    llm-core/             # Foundation types (zero internal deps)
    llm-auth/             # Auth framework
    llm-tools/            # Tool system
    llm-questionnaire/    # Questionnaire engine
    llm-config/           # Typed configuration
    llm-provider-api/     # Provider-facing contracts
    llm-store/            # Storage abstractions
    llm-session/          # Session orchestration
    llm-provider-openai/  # OpenAI provider implementation
    llm-app/              # Application service layer
    llm-cli/              # CLI binary (llmctl)
    llm-gui-api/          # GUI facade
  reference/              # Legacy code that informed the design
```

## License

This project is unlicensed. All rights reserved.
