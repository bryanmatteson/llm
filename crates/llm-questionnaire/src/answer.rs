use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ids::QuestionId;

/// A typed answer value produced by a single question.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value")]
pub enum AnswerValue {
    Choice(String),
    YesNo(bool),
    Text(Option<String>),
    Number(f64),
    MultiSelect(Vec<String>),
}

impl AnswerValue {
    pub fn as_choice(&self) -> Option<&str> {
        match self {
            Self::Choice(v) => Some(v.as_str()),
            _ => None,
        }
    }

    pub fn as_yes_no(&self) -> Option<bool> {
        match self {
            Self::YesNo(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(Some(v)) => Some(v.as_str()),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_multi_select(&self) -> Option<&[String]> {
        match self {
            Self::MultiSelect(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Convert this answer into a [`serde_json::Value`] for condition evaluation.
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            Self::Choice(v) => serde_json::Value::String(v.clone()),
            Self::YesNo(v) => serde_json::Value::Bool(*v),
            Self::Text(Some(v)) => serde_json::Value::String(v.clone()),
            Self::Text(None) => serde_json::Value::Null,
            Self::Number(v) => serde_json::json!(*v),
            Self::MultiSelect(v) => serde_json::json!(v),
        }
    }
}

/// A keyed collection of answers, indexed by [`QuestionId`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnswerMap {
    answers: HashMap<QuestionId, AnswerValue>,
}

impl AnswerMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, id: QuestionId, value: AnswerValue) {
        self.answers.insert(id, value);
    }

    pub fn get(&self, id: &QuestionId) -> Option<&AnswerValue> {
        self.answers.get(id)
    }

    /// Convenience: get a choice string.
    pub fn choice(&self, id: &QuestionId) -> Option<&str> {
        self.answers.get(id).and_then(AnswerValue::as_choice)
    }

    /// Convenience: get a yes/no boolean.
    pub fn yes_no(&self, id: &QuestionId) -> Option<bool> {
        self.answers.get(id).and_then(AnswerValue::as_yes_no)
    }

    /// Convenience: get a text string.
    pub fn text(&self, id: &QuestionId) -> Option<&str> {
        self.answers.get(id).and_then(AnswerValue::as_text)
    }

    /// Convenience: get a number.
    pub fn number(&self, id: &QuestionId) -> Option<f64> {
        self.answers.get(id).and_then(AnswerValue::as_number)
    }

    /// Returns true if an answer exists for the given question id.
    pub fn contains(&self, id: &QuestionId) -> bool {
        self.answers.contains_key(id)
    }

    /// Convert the answer for `id` into a [`serde_json::Value`] (used by condition evaluation).
    pub fn to_json_value(&self, id: &QuestionId) -> Option<serde_json::Value> {
        self.answers.get(id).map(AnswerValue::to_json_value)
    }

    /// Number of recorded answers.
    pub fn len(&self) -> usize {
        self.answers.len()
    }

    /// True when no answers have been recorded.
    pub fn is_empty(&self) -> bool {
        self.answers.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&QuestionId, &AnswerValue)> {
        self.answers.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut map = AnswerMap::new();
        let id = QuestionId::new("q1");
        map.insert(id.clone(), AnswerValue::Choice("a".into()));
        assert_eq!(map.get(&id), Some(&AnswerValue::Choice("a".into())));
    }

    #[test]
    fn typed_accessor_choice() {
        let mut map = AnswerMap::new();
        let id = QuestionId::new("color");
        map.insert(id.clone(), AnswerValue::Choice("red".into()));
        assert_eq!(map.choice(&id), Some("red"));
        assert_eq!(map.yes_no(&id), None);
    }

    #[test]
    fn typed_accessor_yes_no() {
        let mut map = AnswerMap::new();
        let id = QuestionId::new("agree");
        map.insert(id.clone(), AnswerValue::YesNo(false));
        assert_eq!(map.yes_no(&id), Some(false));
        assert_eq!(map.choice(&id), None);
    }

    #[test]
    fn typed_accessor_text() {
        let mut map = AnswerMap::new();
        let id = QuestionId::new("name");
        map.insert(id.clone(), AnswerValue::Text(Some("Alice".into())));
        assert_eq!(map.text(&id), Some("Alice"));

        let id2 = QuestionId::new("empty_text");
        map.insert(id2.clone(), AnswerValue::Text(None));
        assert_eq!(map.text(&id2), None);
    }

    #[test]
    fn typed_accessor_number() {
        let mut map = AnswerMap::new();
        let id = QuestionId::new("count");
        map.insert(id.clone(), AnswerValue::Number(42.5));
        assert_eq!(map.number(&id), Some(42.5));
    }

    #[test]
    fn contains_and_missing() {
        let mut map = AnswerMap::new();
        let id = QuestionId::new("exists");
        map.insert(id.clone(), AnswerValue::YesNo(true));
        assert!(map.contains(&id));
        assert!(!map.contains(&QuestionId::new("nope")));
    }

    #[test]
    fn to_json_value_choice() {
        let mut map = AnswerMap::new();
        let id = QuestionId::new("q");
        map.insert(id.clone(), AnswerValue::Choice("opt".into()));
        assert_eq!(
            map.to_json_value(&id),
            Some(serde_json::Value::String("opt".into()))
        );
    }

    #[test]
    fn to_json_value_yes_no() {
        let mut map = AnswerMap::new();
        let id = QuestionId::new("q");
        map.insert(id.clone(), AnswerValue::YesNo(true));
        assert_eq!(map.to_json_value(&id), Some(serde_json::Value::Bool(true)));
    }

    #[test]
    fn to_json_value_number() {
        let mut map = AnswerMap::new();
        let id = QuestionId::new("q");
        map.insert(id.clone(), AnswerValue::Number(7.0));
        assert_eq!(map.to_json_value(&id), Some(serde_json::json!(7.0)));
    }

    #[test]
    fn to_json_value_multi_select() {
        let mut map = AnswerMap::new();
        let id = QuestionId::new("q");
        map.insert(
            id.clone(),
            AnswerValue::MultiSelect(vec!["a".into(), "b".into()]),
        );
        assert_eq!(map.to_json_value(&id), Some(serde_json::json!(["a", "b"])));
    }

    #[test]
    fn to_json_value_missing() {
        let map = AnswerMap::new();
        assert_eq!(map.to_json_value(&QuestionId::new("x")), None);
    }

    #[test]
    fn len_and_is_empty() {
        let mut map = AnswerMap::new();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        map.insert(QuestionId::new("a"), AnswerValue::YesNo(true));
        assert!(!map.is_empty());
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn answer_value_serde_roundtrip() {
        let values = vec![
            AnswerValue::Choice("opt".into()),
            AnswerValue::YesNo(true),
            AnswerValue::Text(Some("hello".into())),
            AnswerValue::Text(None),
            AnswerValue::Number(1.5),
            AnswerValue::MultiSelect(vec!["a".into(), "b".into()]),
        ];
        for v in &values {
            let json = serde_json::to_string(v).unwrap();
            let back: AnswerValue = serde_json::from_str(&json).unwrap();
            assert_eq!(&back, v);
        }
    }
}
