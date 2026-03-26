use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderCapability {
    OAuth,
    ApiKeyAuth,
    ToolCalling,
    Streaming,
    ModelListing,
    SystemPrompt,
    ImageInput,
    FileInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCapability {
    ToolUse,
    Streaming,
    SystemPrompt,
    Vision,
    LongContext,
}
