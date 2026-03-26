use serde::{Deserialize, Serialize};

use crate::capabilities::{ModelCapability, ProviderCapability};
use crate::ids::{ModelId, ProviderId};
use crate::metadata::Metadata;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderDescriptor {
    pub id: ProviderId,
    pub display_name: String,
    pub default_model: ModelId,
    pub capabilities: Vec<ProviderCapability>,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptor {
    pub id: ModelId,
    pub provider: ProviderId,
    pub display_name: String,
    pub context_window: Option<u64>,
    pub capabilities: Vec<ModelCapability>,
    pub metadata: Metadata,
}
