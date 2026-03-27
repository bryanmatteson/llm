use std::sync::LazyLock;

use llm_core::{FrameworkError, Result};
use regex_lite::Regex;

const MAX_NAME_LEN: usize = 64;
const MAX_DESCRIPTION_LEN: usize = 1024;
const RESERVED_WORDS: &[&str] = &["anthropic", "claude"];

static NAME_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$").unwrap());

/// Validate a skill name according to the Agent Skills specification.
///
/// Rules:
/// - Max 64 characters
/// - Only lowercase letters, numbers, and hyphens
/// - Cannot start or end with a hyphen
/// - Cannot contain reserved words ("anthropic", "claude")
pub fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(FrameworkError::validation("skill name must not be empty"));
    }

    if name.len() > MAX_NAME_LEN {
        return Err(FrameworkError::validation(format!(
            "skill name exceeds {MAX_NAME_LEN} characters: {len}",
            len = name.len()
        )));
    }

    if !NAME_RE.is_match(name) {
        return Err(FrameworkError::validation(format!(
            "skill name must contain only lowercase letters, numbers, and hyphens, \
             and cannot start or end with a hyphen: \"{name}\""
        )));
    }

    for word in RESERVED_WORDS {
        if name.contains(word) {
            return Err(FrameworkError::validation(format!(
                "skill name must not contain reserved word \"{word}\": \"{name}\""
            )));
        }
    }

    Ok(())
}

/// Validate a skill description according to the Agent Skills specification.
///
/// Rules:
/// - Non-empty
/// - Max 1024 characters
/// - Cannot contain XML-like tags (`<` or `>`)
pub fn validate_description(description: &str) -> Result<()> {
    if description.trim().is_empty() {
        return Err(FrameworkError::validation(
            "skill description must not be empty",
        ));
    }

    if description.len() > MAX_DESCRIPTION_LEN {
        return Err(FrameworkError::validation(format!(
            "skill description exceeds {MAX_DESCRIPTION_LEN} characters: {len}",
            len = description.len()
        )));
    }

    if description.contains('<') || description.contains('>') {
        return Err(FrameworkError::validation(
            "skill description must not contain XML tags (< or >)",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- validate_name -------------------------------------------------------

    #[test]
    fn valid_names() {
        for name in ["pdf", "pdf-processing", "my-skill-123", "a", "a1b2"] {
            assert!(validate_name(name).is_ok(), "expected valid: {name}");
        }
    }

    #[test]
    fn empty_name() {
        assert!(validate_name("").is_err());
    }

    #[test]
    fn name_too_long() {
        let long = "a".repeat(MAX_NAME_LEN + 1);
        assert!(validate_name(&long).is_err());
    }

    #[test]
    fn name_at_max_length() {
        let exact = "a".repeat(MAX_NAME_LEN);
        assert!(validate_name(&exact).is_ok());
    }

    #[test]
    fn name_uppercase_rejected() {
        assert!(validate_name("MySkill").is_err());
    }

    #[test]
    fn name_starts_with_hyphen() {
        assert!(validate_name("-pdf").is_err());
    }

    #[test]
    fn name_ends_with_hyphen() {
        assert!(validate_name("pdf-").is_err());
    }

    #[test]
    fn name_contains_spaces() {
        assert!(validate_name("my skill").is_err());
    }

    #[test]
    fn name_contains_underscore() {
        assert!(validate_name("my_skill").is_err());
    }

    #[test]
    fn name_reserved_anthropic() {
        assert!(validate_name("my-anthropic-skill").is_err());
    }

    #[test]
    fn name_reserved_claude() {
        assert!(validate_name("claude-helper").is_err());
    }

    // -- validate_description ------------------------------------------------

    #[test]
    fn valid_description() {
        assert!(validate_description("Processes PDF files and extracts text.").is_ok());
    }

    #[test]
    fn empty_description() {
        assert!(validate_description("").is_err());
    }

    #[test]
    fn whitespace_description_rejected() {
        assert!(validate_description("   \n\t").is_err());
    }

    #[test]
    fn description_too_long() {
        let long = "a".repeat(MAX_DESCRIPTION_LEN + 1);
        assert!(validate_description(&long).is_err());
    }

    #[test]
    fn description_at_max_length() {
        let exact = "a".repeat(MAX_DESCRIPTION_LEN);
        assert!(validate_description(&exact).is_ok());
    }

    #[test]
    fn description_with_xml_tags() {
        assert!(validate_description("Use <tool> for processing").is_err());
        assert!(validate_description("Output > input").is_err());
    }
}
