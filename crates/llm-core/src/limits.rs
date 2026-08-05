use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Controls how a session turns its durable transcript into the message list
/// sent to a provider.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum ContextPolicy {
    /// Send the complete transcript on every provider request.
    #[default]
    FullHistory,

    /// Replace older turns with an auditable structured checkpoint when the
    /// configured input budget is approached.
    Compact(ContextCompaction),
}

/// Token budget and retention settings for [`ContextPolicy::Compact`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextCompaction {
    /// Explicit model context window. When omitted, the provider client may
    /// supply one. Compaction remains dormant if neither source has a value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window_tokens: Option<u64>,

    /// Percentage of the usable input window at which compaction begins.
    /// Values are clamped to 1..=100 when used.
    pub trigger_percent: u8,

    /// Tokens reserved for the provider's response.
    pub reserved_output_tokens: u64,

    /// Minimum number of recent messages retained verbatim. Compaction moves
    /// only across complete user turns, so additional messages may remain.
    pub preserve_recent_messages: usize,

    /// Maximum output requested from the model that creates a checkpoint.
    pub summary_max_tokens: u32,
}

impl Default for ContextCompaction {
    fn default() -> Self {
        Self {
            context_window_tokens: None,
            trigger_percent: 80,
            reserved_output_tokens: 4_096,
            preserve_recent_messages: 12,
            summary_max_tokens: 1_024,
        }
    }
}

/// Resource limits governing a single session's turn loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLimits {
    /// Maximum number of LLM turns (each containing potential tool calls and
    /// a follow-up response) before the loop is forcibly terminated.
    pub max_turns: usize,

    /// Maximum number of tool calls the mediator will execute within a single
    /// turn before returning control.
    pub max_tool_calls_per_turn: usize,

    /// How long to wait for the provider to return a complete turn response
    /// before timing out.
    #[serde(with = "duration_secs")]
    pub turn_timeout: Duration,

    /// How long to wait for a single tool execution before timing out.
    #[serde(with = "duration_secs")]
    pub tool_timeout: Duration,

    /// Policy for projecting the durable transcript into provider context.
    #[serde(default)]
    pub context: ContextPolicy,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_turns: 10,
            max_tool_calls_per_turn: 8,
            turn_timeout: Duration::from_secs(120),
            tool_timeout: Duration::from_secs(30),
            context: ContextPolicy::default(),
        }
    }
}

/// Serde helper: serialize/deserialize `Duration` as an integer number of
/// seconds. This keeps the JSON representation human-readable.
mod duration_secs {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(duration.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
        let secs = u64::deserialize(deserializer)?;
        Ok(Duration::from_secs(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let limits = SessionLimits::default();
        assert_eq!(limits.max_turns, 10);
        assert_eq!(limits.max_tool_calls_per_turn, 8);
        assert_eq!(limits.turn_timeout, Duration::from_secs(120));
        assert_eq!(limits.tool_timeout, Duration::from_secs(30));
        assert!(matches!(limits.context, ContextPolicy::FullHistory));
    }

    #[test]
    fn serde_roundtrip() {
        let limits = SessionLimits::default();
        let json = serde_json::to_string(&limits).unwrap();
        let back: SessionLimits = serde_json::from_str(&json).unwrap();
        assert_eq!(back.max_turns, limits.max_turns);
        assert_eq!(back.max_tool_calls_per_turn, limits.max_tool_calls_per_turn);
        assert_eq!(back.turn_timeout, limits.turn_timeout);
        assert_eq!(back.tool_timeout, limits.tool_timeout);
    }

    #[test]
    fn compact_policy_serde_roundtrip() {
        let policy = ContextPolicy::Compact(ContextCompaction {
            context_window_tokens: Some(128_000),
            ..ContextCompaction::default()
        });
        let json = serde_json::to_string(&policy).unwrap();
        let back: ContextPolicy = serde_json::from_str(&json).unwrap();
        match back {
            ContextPolicy::Compact(config) => {
                assert_eq!(config.context_window_tokens, Some(128_000));
            }
            ContextPolicy::FullHistory => panic!("expected compact policy"),
        }
    }

    #[test]
    fn compact_policy_uses_defaults_for_omitted_settings() {
        let policy: ContextPolicy =
            serde_json::from_str(r#"{"strategy":"compact","context_window_tokens":32000}"#)
                .unwrap();
        let ContextPolicy::Compact(config) = policy else {
            panic!("expected compact policy");
        };
        assert_eq!(config.context_window_tokens, Some(32_000));
        assert_eq!(config.trigger_percent, 80);
        assert_eq!(config.preserve_recent_messages, 12);
    }
}
