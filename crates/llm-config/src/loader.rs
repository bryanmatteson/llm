use std::path::Path;

use llm_core::{FrameworkError, Result};

use crate::app::AppConfig;

pub struct ConfigLoader;

impl ConfigLoader {
    pub fn load_from_file(path: &Path) -> Result<AppConfig> {
        let contents = std::fs::read_to_string(path).map_err(|e| {
            FrameworkError::config(format!("failed to read config file {}: {e}", path.display()))
        })?;
        let config: AppConfig = toml::from_str(&contents).map_err(|e| {
            FrameworkError::config(format!(
                "failed to parse config file {}: {e}",
                path.display()
            ))
        })?;
        Ok(config)
    }

    pub fn save_to_file(config: &AppConfig, path: &Path) -> Result<()> {
        let contents = toml::to_string_pretty(config).map_err(|e| {
            FrameworkError::config(format!("failed to serialize config: {e}"))
        })?;
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
