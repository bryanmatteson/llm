use std::sync::Arc;

use llm_core::SkillId;
use llm_skill::{Skill, SkillRegistry};

/// High-level service for querying the skill registry.
pub struct SkillService {
    registry: Arc<SkillRegistry>,
}

impl SkillService {
    pub fn new(registry: Arc<SkillRegistry>) -> Self {
        Self { registry }
    }

    /// List all registered skills in sorted order.
    pub fn list_skills(&self) -> Vec<&Skill> {
        self.registry.list()
    }

    /// Look up a skill by its ID.
    pub fn get_skill(&self, id: &SkillId) -> Option<&Skill> {
        self.registry.get(id)
    }

    /// Look up a skill by its name string.
    pub fn get_skill_by_name(&self, name: &str) -> Option<&Skill> {
        self.registry.get_by_name(name)
    }

    /// Generate a metadata summary for system prompt injection.
    ///
    /// Returns `None` if no skills are registered.
    pub fn metadata_prompt(&self) -> Option<String> {
        self.registry.metadata_prompt()
    }

    /// Obtain a shared reference to the underlying [`SkillRegistry`].
    pub fn registry(&self) -> &Arc<SkillRegistry> {
        &self.registry
    }

    /// Returns the number of registered skills.
    pub fn len(&self) -> usize {
        self.registry.len()
    }

    /// Returns `true` if no skills are registered.
    pub fn is_empty(&self) -> bool {
        self.registry.is_empty()
    }
}
