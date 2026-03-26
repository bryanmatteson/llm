use std::time::Duration;

use serde::{Deserialize, Serialize};

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
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_turns: 10,
            max_tool_calls_per_turn: 8,
            turn_timeout: Duration::from_secs(120),
            tool_timeout: Duration::from_secs(30),
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
}
