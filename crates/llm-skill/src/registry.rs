use std::collections::BTreeMap;
use std::path::Path;

use llm_core::{Result, SkillId};

use crate::loader::{Skill, discover_skills, load_skill};

/// A registry of discovered skills.
///
/// Stores skills in sorted order (by ID) for deterministic iteration.
/// Provides methods for lookup, listing, and generating system prompt metadata.
#[derive(Debug, Clone)]
pub struct SkillRegistry {
    skills: BTreeMap<SkillId, Skill>,
}

impl SkillRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            skills: BTreeMap::new(),
        }
    }

    /// Discover and load skills from multiple directories.
    ///
    /// Each directory is scanned recursively for nested `SKILL.md` files.
    /// Duplicate skill IDs are resolved by last-write-wins (later directories
    /// override earlier ones).
    ///
    /// Individual skill loading errors are silently ignored — use
    /// [`discover_skills`] directly for fine-grained error handling.
    pub fn discover(dirs: &[impl AsRef<Path>]) -> Self {
        Self::discover_with_errors(dirs).0
    }

    /// Discover and load skills from multiple directories, returning any
    /// individual load failures alongside the resulting registry.
    pub fn discover_with_errors(
        dirs: &[impl AsRef<Path>],
    ) -> (Self, Vec<(std::path::PathBuf, llm_core::FrameworkError)>) {
        let mut registry = Self::new();
        let mut errors = Vec::new();
        for dir in dirs {
            let (skills, mut dir_errors) = discover_skills(dir.as_ref());
            for skill in skills {
                registry.register(skill);
            }
            errors.append(&mut dir_errors);
        }
        (registry, errors)
    }

    /// Register a pre-loaded skill.
    ///
    /// If a skill with the same ID already exists, it is replaced.
    pub fn register(&mut self, skill: Skill) {
        self.skills.insert(skill.id.clone(), skill);
    }

    /// Load a skill from a directory and register it.
    pub fn load_and_register(&mut self, skill_dir: &Path) -> Result<()> {
        let skill = load_skill(skill_dir)?;
        self.register(skill);
        Ok(())
    }

    /// Look up a skill by its ID.
    pub fn get(&self, id: &SkillId) -> Option<&Skill> {
        self.skills.get(id)
    }

    /// Look up a skill by its name string.
    pub fn get_by_name(&self, name: &str) -> Option<&Skill> {
        self.skills.get(&SkillId::new(name))
    }

    /// Return all registered skills in sorted order.
    pub fn list(&self) -> Vec<&Skill> {
        self.skills.values().collect()
    }

    /// Return all skill IDs in sorted order.
    pub fn skill_ids(&self) -> Vec<SkillId> {
        self.skills.keys().cloned().collect()
    }

    /// Returns the number of registered skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Returns `true` if the registry contains no skills.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Generate a metadata summary suitable for system prompt injection.
    ///
    /// Returns `None` if the registry is empty. Otherwise returns a
    /// formatted string listing each skill's name and description.
    ///
    /// This is the Level 1 (metadata-only) representation — the skill
    /// instructions body is not included.
    pub fn metadata_prompt(&self) -> Option<String> {
        if self.skills.is_empty() {
            return None;
        }

        let mut lines = vec![
            "The following skills are available:".to_string(),
            String::new(),
        ];

        for skill in self.skills.values() {
            let description = skill
                .metadata
                .description
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            lines.push(format!("- {}: {}", skill.metadata.name, description));
        }

        Some(lines.join("\n"))
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llm-skill-reg-{}", std::process::id()));
        let dir = dir.join(format!("{}", fastrand::u64(..)));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_skill_dir(parent: &Path, name: &str, desc: &str) -> PathBuf {
        let dir = parent.join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {desc}\n---\n\nInstructions for {name}.\n"),
        )
        .unwrap();
        dir
    }

    #[test]
    fn empty_registry() {
        let reg = SkillRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(reg.list().is_empty());
        assert!(reg.metadata_prompt().is_none());
    }

    #[test]
    fn register_and_lookup() {
        let tmp = tempdir();
        let dir = make_skill_dir(&tmp, "test-skill", "A test skill.");
        let skill = load_skill(&dir).unwrap();

        let mut reg = SkillRegistry::new();
        reg.register(skill);

        assert_eq!(reg.len(), 1);
        assert!(reg.get(&SkillId::new("test-skill")).is_some());
        assert!(reg.get_by_name("test-skill").is_some());
        assert!(reg.get_by_name("nonexistent").is_none());
    }

    #[test]
    fn discover_from_directories() {
        let dir_a = tempdir();
        let dir_b = tempdir();

        make_skill_dir(&dir_a, "alpha", "Alpha skill.");
        make_skill_dir(&dir_b, "beta", "Beta skill.");

        let reg = SkillRegistry::discover(&[&dir_a, &dir_b]);
        assert_eq!(reg.len(), 2);

        let ids = reg.skill_ids();
        assert_eq!(ids[0].as_str(), "alpha");
        assert_eq!(ids[1].as_str(), "beta");
    }

    #[test]
    fn duplicate_id_last_wins() {
        let dir_a = tempdir();
        let dir_b = tempdir();

        make_skill_dir(&dir_a, "shared", "Version A.");
        make_skill_dir(&dir_b, "shared", "Version B.");

        let reg = SkillRegistry::discover(&[&dir_a, &dir_b]);
        assert_eq!(reg.len(), 1);

        let skill = reg.get_by_name("shared").unwrap();
        assert_eq!(skill.metadata.description, "Version B.");
    }

    #[test]
    fn discover_with_errors_collects_invalid_skills() {
        let tmp = tempdir();
        make_skill_dir(&tmp, "alpha", "Alpha skill.");

        let bad_dir = tmp.join("bad");
        fs::create_dir_all(&bad_dir).unwrap();
        fs::write(
            bad_dir.join("SKILL.md"),
            "---\nname: BAD\ndescription: Broken skill.\n---\n",
        )
        .unwrap();

        let (reg, errors) = SkillRegistry::discover_with_errors(&[&tmp]);
        assert_eq!(reg.len(), 1);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn metadata_prompt_format() {
        let tmp = tempdir();
        make_skill_dir(&tmp, "alpha", "Does alpha things.");
        make_skill_dir(&tmp, "beta", "Does beta things.");

        let reg = SkillRegistry::discover(&[&tmp]);
        let prompt = reg.metadata_prompt().unwrap();

        assert!(prompt.starts_with("The following skills are available:"));
        assert!(prompt.contains("- alpha: Does alpha things."));
        assert!(prompt.contains("- beta: Does beta things."));
    }

    #[test]
    fn sorted_iteration_order() {
        let tmp = tempdir();
        make_skill_dir(&tmp, "zulu", "Last.");
        make_skill_dir(&tmp, "alpha", "First.");
        make_skill_dir(&tmp, "mike", "Middle.");

        let reg = SkillRegistry::discover(&[&tmp]);
        let names: Vec<&str> = reg
            .list()
            .iter()
            .map(|s| s.metadata.name.as_str())
            .collect();
        assert_eq!(names, vec!["alpha", "mike", "zulu"]);
    }
}
