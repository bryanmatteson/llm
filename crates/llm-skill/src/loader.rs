use std::fs;
use std::path::{Path, PathBuf};

use llm_core::{FrameworkError, Result, SkillId};
use serde::Deserialize;

use crate::validate::{validate_description, validate_name};

/// The parsed content of a `SKILL.md` file.
#[derive(Debug, Clone)]
pub struct SkillMetadata {
    /// The skill name from frontmatter (kebab-case).
    pub name: String,
    /// The skill description from frontmatter.
    pub description: String,
}

/// A fully loaded skill.
#[derive(Debug, Clone)]
pub struct Skill {
    /// Framework-level identifier derived from the skill name.
    pub id: SkillId,
    /// Parsed metadata from the YAML frontmatter.
    pub metadata: SkillMetadata,
    /// The markdown body of SKILL.md (Level 2 instructions).
    pub instructions: String,
    /// The directory containing the skill (for Level 3 file access).
    pub path: PathBuf,
}

const SKILL_FILE: &str = "SKILL.md";

/// Parse a `SKILL.md` file at the given path.
///
/// Returns a [`Skill`] with validated metadata and instructions body.
/// The `skill_dir` is recorded as the skill's base path for Level 3 access.
pub fn load_skill(skill_dir: &Path) -> Result<Skill> {
    let file_path = skill_dir.join(SKILL_FILE);
    let content = fs::read_to_string(&file_path).map_err(|e| {
        FrameworkError::config(format!("failed to read {}: {e}", file_path.display()))
    })?;

    let (metadata, instructions) = parse_frontmatter(&content).map_err(|e| {
        FrameworkError::config(format!("invalid SKILL.md at {}: {e}", file_path.display()))
    })?;

    validate_name(&metadata.name)?;
    validate_description(&metadata.description)?;

    let id = SkillId::new(metadata.name.as_str());

    Ok(Skill {
        id,
        metadata,
        instructions,
        path: skill_dir.to_path_buf(),
    })
}

/// Discover skills in a directory.
///
/// Recursively scans `dir` for nested directories containing a `SKILL.md`
/// file. Also checks if `dir` itself contains a `SKILL.md` (for flat
/// layouts); when it does, that directory is treated as a single skill root.
///
/// Returns all successfully loaded skills. Errors for individual skills are
/// collected and returned alongside the successful ones.
pub fn discover_skills(dir: &Path) -> (Vec<Skill>, Vec<(PathBuf, FrameworkError)>) {
    let mut skills = Vec::new();
    let mut errors = Vec::new();

    if !dir.exists() {
        return (skills, errors);
    }

    discover_skills_inner(dir, &mut skills, &mut errors);
    (skills, errors)
}

fn discover_skills_inner(
    dir: &Path,
    skills: &mut Vec<Skill>,
    errors: &mut Vec<(PathBuf, FrameworkError)>,
) {
    if dir.join(SKILL_FILE).is_file() {
        match load_skill(dir) {
            Ok(skill) => skills.push(skill),
            Err(e) => errors.push((dir.to_path_buf(), e)),
        }
        return;
    }

    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            errors.push((
                dir.to_path_buf(),
                FrameworkError::config(format!("failed to read directory {}: {e}", dir.display())),
            ));
            return;
        }
    };

    let mut paths: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    paths.sort();

    for path in paths {
        discover_skills_inner(&path, skills, errors);
    }
}

// ---------------------------------------------------------------------------
// Frontmatter parser
// ---------------------------------------------------------------------------

/// Parse YAML frontmatter from a `SKILL.md` string.
///
/// The expected format is:
///
/// ```text
/// ---
/// name: skill-name
/// description: Brief description of the skill
/// ---
///
/// # Instructions body...
/// ```
fn parse_frontmatter(content: &str) -> std::result::Result<(SkillMetadata, String), String> {
    #[derive(Deserialize)]
    struct Frontmatter {
        name: String,
        description: String,
    }

    let trimmed = content.trim_start();

    if !trimmed.starts_with("---") {
        return Err("missing opening --- frontmatter delimiter".into());
    }

    let after_open = &trimmed[3..];
    let after_open = strip_single_newline(after_open)
        .ok_or("opening --- frontmatter delimiter must be followed by a newline")?;

    let (frontmatter_str, body) =
        split_frontmatter_block(after_open).ok_or("missing closing --- frontmatter delimiter")?;

    let frontmatter: Frontmatter = serde_yaml::from_str(frontmatter_str)
        .map_err(|e| format!("invalid YAML frontmatter: {e}"))?;

    Ok((
        SkillMetadata {
            name: frontmatter.name,
            description: frontmatter.description,
        },
        body.trim().to_string(),
    ))
}

fn split_frontmatter_block(content: &str) -> Option<(&str, &str)> {
    let mut offset = 0;
    for segment in content.split_inclusive('\n') {
        let line = segment
            .strip_suffix('\n')
            .unwrap_or(segment)
            .strip_suffix('\r')
            .unwrap_or(segment.strip_suffix('\n').unwrap_or(segment));
        if line == "---" {
            let body = if offset + segment.len() < content.len() {
                let rest = &content[offset + segment.len()..];
                strip_single_newline(rest).unwrap_or(rest)
            } else {
                ""
            };
            return Some((&content[..offset], body));
        }
        offset += segment.len();
    }

    if content == "---" {
        return Some(("", ""));
    }

    None
}

fn strip_single_newline(s: &str) -> Option<&str> {
    s.strip_prefix("\r\n").or_else(|| s.strip_prefix('\n'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn make_skill_dir(tmp: &Path, name: &str, content: &str) -> PathBuf {
        let dir = tmp.join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("SKILL.md"), content).unwrap();
        dir
    }

    // -- parse_frontmatter ---------------------------------------------------

    #[test]
    fn parse_valid_frontmatter() {
        let content = "\
---
name: pdf-processing
description: Extract text and tables from PDF files.
---

# PDF Processing

Use pdfplumber for extraction.
";
        let (meta, body) = parse_frontmatter(content).unwrap();
        assert_eq!(meta.name, "pdf-processing");
        assert_eq!(meta.description, "Extract text and tables from PDF files.");
        assert!(body.starts_with("# PDF Processing"));
        assert!(body.contains("pdfplumber"));
    }

    #[test]
    fn parse_quoted_values() {
        let content = "\
---
name: \"my-skill\"
description: 'A skill with quotes'
---

Body here.
";
        let (meta, _body) = parse_frontmatter(content).unwrap();
        assert_eq!(meta.name, "my-skill");
        assert_eq!(meta.description, "A skill with quotes");
    }

    #[test]
    fn parse_yaml_multiline_description() {
        let content = "\
---
name: yaml-skill
description: >
  A skill with
  folded YAML text.
---

Body here.
";
        let (meta, body) = parse_frontmatter(content).unwrap();
        assert_eq!(meta.name, "yaml-skill");
        assert_eq!(meta.description, "A skill with folded YAML text.\n");
        assert_eq!(body, "Body here.");
    }

    #[test]
    fn parse_empty_body() {
        let content = "\
---
name: minimal
description: A minimal skill
---
";
        let (meta, body) = parse_frontmatter(content).unwrap();
        assert_eq!(meta.name, "minimal");
        assert!(body.is_empty());
    }

    #[test]
    fn parse_missing_opening_delimiter() {
        let content = "name: bad\n---\n";
        assert!(parse_frontmatter(content).is_err());
    }

    #[test]
    fn parse_missing_closing_delimiter() {
        let content = "---\nname: bad\n";
        assert!(parse_frontmatter(content).is_err());
    }

    #[test]
    fn parse_missing_name() {
        let content = "---\ndescription: No name\n---\n";
        assert!(parse_frontmatter(content).is_err());
    }

    #[test]
    fn parse_missing_description() {
        let content = "---\nname: no-desc\n---\n";
        assert!(parse_frontmatter(content).is_err());
    }

    // -- CRLF support --------------------------------------------------------

    #[test]
    fn parse_crlf_line_endings() {
        let content =
            "---\r\nname: crlf-skill\r\ndescription: A CRLF skill.\r\n---\r\n\r\n# Hello CRLF\r\n";
        let (meta, body) = parse_frontmatter(content).unwrap();
        assert_eq!(meta.name, "crlf-skill");
        assert_eq!(meta.description, "A CRLF skill.");
        assert!(body.contains("# Hello CRLF"));
    }

    #[test]
    fn parse_crlf_empty_body() {
        let content = "---\r\nname: crlf-min\r\ndescription: Minimal CRLF.\r\n---\r\n";
        let (meta, body) = parse_frontmatter(content).unwrap();
        assert_eq!(meta.name, "crlf-min");
        assert!(body.is_empty());
    }

    #[test]
    fn parse_mixed_line_endings() {
        let content = "---\nname: mixed\r\ndescription: Mixed endings.\n---\nBody.\n";
        let (meta, body) = parse_frontmatter(content).unwrap();
        assert_eq!(meta.name, "mixed");
        assert!(body.contains("Body."));
    }

    // -- delimiter edge cases ------------------------------------------------

    #[test]
    fn parse_four_dashes_opening_rejected() {
        let content = "----\nname: bad\ndescription: Bad.\n---\n";
        assert!(parse_frontmatter(content).is_err());
    }

    #[test]
    fn parse_dashes_in_body_not_treated_as_delimiter() {
        let content = "---\nname: my-skill\ndescription: A skill.\n---\n\n---extra text here---\n\nMore body.\n";
        let (meta, body) = parse_frontmatter(content).unwrap();
        assert_eq!(meta.name, "my-skill");
        assert!(body.contains("---extra text here---"));
        assert!(body.contains("More body."));
    }

    #[test]
    fn parse_description_with_colons() {
        let content =
            "---\nname: my-skill\ndescription: \"Processes files: PDFs, at 12:30.\"\n---\nBody.\n";
        let (meta, _body) = parse_frontmatter(content).unwrap();
        assert_eq!(meta.description, "Processes files: PDFs, at 12:30.");
    }

    // -- load_skill ----------------------------------------------------------

    #[test]
    fn load_valid_skill() {
        let tmp = tempdir();
        let dir = make_skill_dir(
            &tmp,
            "my-skill",
            "---\nname: my-skill\ndescription: A test skill.\n---\n\n# Hello\n",
        );
        let skill = load_skill(&dir).unwrap();
        assert_eq!(skill.id.as_str(), "my-skill");
        assert_eq!(skill.metadata.name, "my-skill");
        assert_eq!(skill.instructions, "# Hello");
        assert_eq!(skill.path, dir);
    }

    #[test]
    fn load_skill_invalid_name() {
        let tmp = tempdir();
        let dir = make_skill_dir(
            &tmp,
            "bad",
            "---\nname: BAD_NAME\ndescription: A skill.\n---\n",
        );
        assert!(load_skill(&dir).is_err());
    }

    #[test]
    fn load_skill_missing_file() {
        let tmp = tempdir();
        let dir = tmp.join("nonexistent");
        fs::create_dir_all(&dir).unwrap();
        assert!(load_skill(&dir).is_err());
    }

    // -- discover_skills -----------------------------------------------------

    #[test]
    fn discover_in_directory() {
        let tmp = tempdir();
        make_skill_dir(
            &tmp,
            "alpha",
            "---\nname: alpha\ndescription: Alpha skill.\n---\nBody A",
        );
        make_skill_dir(
            &tmp,
            "beta",
            "---\nname: beta\ndescription: Beta skill.\n---\nBody B",
        );
        // Invalid skill (bad name) — should appear in errors
        make_skill_dir(
            &tmp,
            "bad",
            "---\nname: BAD\ndescription: Bad skill.\n---\n",
        );

        let (skills, errors) = discover_skills(&tmp);
        assert_eq!(skills.len(), 2);
        assert_eq!(errors.len(), 1);

        let names: Vec<&str> = skills.iter().map(|s| s.metadata.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }

    #[test]
    fn discover_flat_layout() {
        let tmp = tempdir();
        fs::write(
            tmp.join("SKILL.md"),
            "---\nname: flat-skill\ndescription: A flat skill.\n---\nFlat body",
        )
        .unwrap();

        let (skills, errors) = discover_skills(&tmp);
        assert_eq!(skills.len(), 1);
        assert!(errors.is_empty());
        assert_eq!(skills[0].id.as_str(), "flat-skill");
    }

    #[test]
    fn discover_nested_directory() {
        let tmp = tempdir();
        make_skill_dir(
            &tmp.join(".system"),
            "openai-docs",
            "---\nname: openai-docs\ndescription: OpenAI docs helper.\n---\nBody",
        );

        let (skills, errors) = discover_skills(&tmp);
        assert!(errors.is_empty());
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].id.as_str(), "openai-docs");
    }

    #[test]
    fn discover_empty_directory() {
        let tmp = tempdir();
        let (skills, errors) = discover_skills(&tmp);
        assert!(skills.is_empty());
        assert!(errors.is_empty());
    }

    #[test]
    fn discover_nonexistent_directory() {
        let (skills, errors) = discover_skills(Path::new("/nonexistent/path"));
        assert!(skills.is_empty());
        assert!(errors.is_empty());
    }

    // -- test helpers --------------------------------------------------------

    fn tempdir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("llm-skill-test-{}", std::process::id()));
        let dir = dir.join(format!("{}", fastrand::u64(..)));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
