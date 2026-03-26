use std::path::Path;

use llm_core::{FrameworkError, Result};

use crate::app::LlmConfig;

/// Loads [`AppConfig`] from KDL files.
///
/// This is a pure utility — it takes explicit paths and never assumes a
/// default location.  The *application* (CLI binary, GUI shell, etc.)
/// decides where config files live; the library only reads and writes.
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load an [`AppConfig`] from a KDL file at `path`.
    pub fn load_from_file(path: &Path) -> Result<LlmConfig> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            FrameworkError::config(format!("failed to read config file {}: {e}", path.display()))
        })?;
        Self::parse(&contents)
    }

    /// Parse an [`AppConfig`] from a KDL string.
    pub fn parse(kdl: &str) -> Result<LlmConfig> {
        kdl_config::parse_str(kdl)
            .map_err(|e| FrameworkError::config(format!("failed to parse KDL config: {e}")))
    }

    /// Try to load from `path`, returning `Ok(None)` if the file does not
    /// exist.
    pub fn load_optional(path: &Path) -> Result<Option<LlmConfig>> {
        if path.exists() {
            Ok(Some(Self::load_from_file(path)?))
        } else {
            Ok(None)
        }
    }

    /// Render `config` to a KDL string and write it to `path`, creating
    /// parent directories as needed.
    pub fn save_to_file(config: &LlmConfig, path: &Path) -> Result<()> {
        let contents = kdl_config::to_kdl(config, "llm");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                FrameworkError::config(format!(
                    "failed to create config directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        std::fs::write(path, contents).map_err(|e| {
            FrameworkError::config(format!(
                "failed to write config file {}: {e}",
                path.display()
            ))
        })?;
        Ok(())
    }
}
