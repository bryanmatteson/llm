use serde_json::Value;

use llm_core::{FrameworkError, Result};

use crate::tool::ToolDescriptor;

/// Perform basic JSON-schema-style validation of `input` against the parameter
/// schema declared in `descriptor`.
///
/// This is intentionally a lightweight check -- it validates:
///
/// 1. That every field listed in the schema's `"required"` array is present in
///    `input`.
/// 2. That for each property declared in `"properties"`, if the value is
///    present, its JSON type matches the declared `"type"` string.
///
/// It does **not** implement the full JSON Schema specification (e.g. no
/// pattern, enum, oneOf, etc.).
pub fn validate_tool_input(descriptor: &ToolDescriptor, input: &Value) -> Result<()> {
    let schema = &descriptor.parameters;

    // The input must be a JSON object.
    let input_obj = input
        .as_object()
        .ok_or_else(|| FrameworkError::validation("tool input must be a JSON object"))?;

    // ---- required fields ----
    if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
        for req in required {
            if let Some(name) = req.as_str() {
                if !input_obj.contains_key(name) {
                    return Err(FrameworkError::validation(format!(
                        "missing required field: {name}"
                    )));
                }
            }
        }
    }

    // ---- type checks for declared properties ----
    if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
        for (prop_name, prop_schema) in properties {
            if let Some(value) = input_obj.get(prop_name) {
                if let Some(expected_type) = prop_schema.get("type").and_then(|t| t.as_str()) {
                    if !json_type_matches(value, expected_type) {
                        return Err(FrameworkError::validation(format!(
                            "field \"{prop_name}\" expected type \"{expected_type}\", got {}",
                            json_type_name(value),
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

/// Check whether a JSON `value` matches the given JSON-Schema type name.
fn json_type_matches(value: &Value, expected: &str) -> bool {
    match expected {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true, // unknown type: don't reject
    }
}

/// Return a human-readable name for the JSON type of `value`.
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
