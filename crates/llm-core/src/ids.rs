use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

macro_rules! define_id {
    ($name:ident) => {
        #[derive(Clone, PartialEq, Eq, Hash)]
        pub struct $name(Arc<str>);

        impl $name {
            pub fn new(s: impl Into<Arc<str>>) -> Self {
                Self(s.into())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}(\"{}\")", stringify!($name), self.0)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.into())
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s.into())
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.0.cmp(&other.0)
            }
        }

        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.0.serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                let s = String::deserialize(deserializer)?;
                Ok(Self(s.into()))
            }
        }
    };
}

define_id!(ProviderId);
define_id!(ModelId);
define_id!(SessionId);
define_id!(ToolId);
define_id!(QuestionnaireId);
define_id!(QuestionId);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_roundtrip_serde() {
        let id = ProviderId::new("openai");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"openai\"");
        let back: ProviderId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn id_display_and_debug() {
        let id = ModelId::new("gpt-4o");
        assert_eq!(id.to_string(), "gpt-4o");
        assert_eq!(format!("{id:?}"), "ModelId(\"gpt-4o\")");
    }

    #[test]
    fn id_ordering() {
        let a = ToolId::new("alpha");
        let b = ToolId::new("beta");
        assert!(a < b);
    }

    #[test]
    fn id_from_string_and_str() {
        let from_str: SessionId = "session-1".into();
        let from_string: SessionId = String::from("session-1").into();
        assert_eq!(from_str, from_string);
    }
}
