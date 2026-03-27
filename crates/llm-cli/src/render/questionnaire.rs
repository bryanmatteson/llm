use std::io::{self, BufRead, Write};

use llm_core::Result;
use llm_questionnaire::{AnswerMap, AnswerValue, QuestionKind, Questionnaire, QuestionnaireRun};

/// Drive a [`Questionnaire`] interactively on the terminal, returning the
/// collected answers.
pub fn run_terminal_questionnaire(questionnaire: &Questionnaire) -> Result<AnswerMap> {
    let mut run = QuestionnaireRun::new(questionnaire.clone())
        .map_err(|errs| llm_core::FrameworkError::questionnaire(errs.join("; ")))?;

    let stdin = io::stdin();
    let mut reader = stdin.lock();

    while let Some(question) = run.next_question() {
        // Display the question.
        if let Some(help) = &question.help_text {
            eprintln!("  ({help})");
        }

        let answer = match &question.kind {
            QuestionKind::Choice { options, default } => {
                prompt_choice(&question.label, options, default.as_deref(), &mut reader)?
            }
            QuestionKind::YesNo { default } => {
                prompt_yes_no(&question.label, *default, &mut reader)?
            }
            QuestionKind::Text {
                placeholder,
                default,
            } => prompt_text(
                &question.label,
                placeholder.as_deref(),
                default.as_deref(),
                &mut reader,
            )?,
            QuestionKind::Number { min, max, default } => {
                prompt_number(&question.label, *min, *max, *default, &mut reader)?
            }
            QuestionKind::MultiSelect { options, default } => {
                prompt_multi_select(&question.label, options, default.as_deref(), &mut reader)?
            }
        };

        match run.submit_answer(answer) {
            Ok(()) => {}
            Err(errors) => {
                for err in &errors {
                    eprintln!("  Validation error: {err}");
                }
                // The engine did not advance; the question will be re-asked
                // on the next iteration.
            }
        }
    }

    Ok(run.answers().clone())
}

// ---------------------------------------------------------------------------
// Per-kind prompt helpers
// ---------------------------------------------------------------------------

fn prompt_choice(
    label: &str,
    options: &[llm_questionnaire::ChoiceOption],
    default: Option<&str>,
    reader: &mut impl BufRead,
) -> Result<AnswerValue> {
    loop {
        eprintln!("{label}");
        for (i, opt) in options.iter().enumerate() {
            let marker = if default == Some(opt.value.as_str()) {
                " (default)"
            } else {
                ""
            };
            eprintln!("  {}: {}{}", i + 1, opt.label, marker);
            if let Some(desc) = &opt.description {
                eprintln!("       {desc}");
            }
        }

        let default_hint = default.map(|d| format!(" [{d}]")).unwrap_or_default();
        eprint!("Choice{default_hint}: ");
        io::stderr().flush().ok();

        let mut input = String::new();
        reader
            .read_line(&mut input)
            .map_err(|e| llm_core::FrameworkError::questionnaire(format!("read error: {e}")))?;
        let trimmed = input.trim();

        // Empty input -> use default if available.
        if trimmed.is_empty() {
            if let Some(d) = default {
                return Ok(AnswerValue::Choice(d.to_owned()));
            }
            eprintln!("  Please enter a selection.");
            continue;
        }

        // Accept by number.
        if let Ok(num) = trimmed.parse::<usize>() {
            if num >= 1 && num <= options.len() {
                return Ok(AnswerValue::Choice(options[num - 1].value.clone()));
            }
        }

        // Accept by value string.
        if let Some(opt) = options.iter().find(|o| o.value == trimmed) {
            return Ok(AnswerValue::Choice(opt.value.clone()));
        }

        eprintln!("  Invalid selection. Try again.");
    }
}

fn prompt_yes_no(
    label: &str,
    default: Option<bool>,
    reader: &mut impl BufRead,
) -> Result<AnswerValue> {
    loop {
        let hint = match default {
            Some(true) => " [Y/n]",
            Some(false) => " [y/N]",
            None => " [y/n]",
        };
        eprint!("{label}{hint}: ");
        io::stderr().flush().ok();

        let mut input = String::new();
        reader
            .read_line(&mut input)
            .map_err(|e| llm_core::FrameworkError::questionnaire(format!("read error: {e}")))?;
        let trimmed = input.trim().to_lowercase();

        if trimmed.is_empty() {
            if let Some(d) = default {
                return Ok(AnswerValue::YesNo(d));
            }
            eprintln!("  Please enter y or n.");
            continue;
        }

        match trimmed.as_str() {
            "y" | "yes" => return Ok(AnswerValue::YesNo(true)),
            "n" | "no" => return Ok(AnswerValue::YesNo(false)),
            _ => {
                eprintln!("  Please enter y or n.");
            }
        }
    }
}

fn prompt_text(
    label: &str,
    placeholder: Option<&str>,
    default: Option<&str>,
    reader: &mut impl BufRead,
) -> Result<AnswerValue> {
    let hint = placeholder.map(|p| format!(" ({p})")).unwrap_or_default();
    let default_hint = default.map(|d| format!(" [{d}]")).unwrap_or_default();
    eprint!("{label}{hint}{default_hint}: ");
    io::stderr().flush().ok();

    let mut input = String::new();
    reader
        .read_line(&mut input)
        .map_err(|e| llm_core::FrameworkError::questionnaire(format!("read error: {e}")))?;
    let trimmed = input.trim();

    if trimmed.is_empty() {
        if let Some(d) = default {
            Ok(AnswerValue::Text(Some(d.to_owned())))
        } else {
            Ok(AnswerValue::Text(None))
        }
    } else {
        Ok(AnswerValue::Text(Some(trimmed.to_owned())))
    }
}

fn prompt_number(
    label: &str,
    min: Option<f64>,
    max: Option<f64>,
    default: Option<f64>,
    reader: &mut impl BufRead,
) -> Result<AnswerValue> {
    loop {
        let range_hint = match (min, max) {
            (Some(lo), Some(hi)) => format!(" [{lo}..{hi}]"),
            (Some(lo), None) => format!(" [>={lo}]"),
            (None, Some(hi)) => format!(" [<={hi}]"),
            (None, None) => String::new(),
        };
        let default_hint = default
            .map(|d| format!(" (default: {d})"))
            .unwrap_or_default();
        eprint!("{label}{range_hint}{default_hint}: ");
        io::stderr().flush().ok();

        let mut input = String::new();
        reader
            .read_line(&mut input)
            .map_err(|e| llm_core::FrameworkError::questionnaire(format!("read error: {e}")))?;
        let trimmed = input.trim();

        if trimmed.is_empty() {
            if let Some(d) = default {
                return Ok(AnswerValue::Number(d));
            }
            eprintln!("  Please enter a number.");
            continue;
        }

        match trimmed.parse::<f64>() {
            Ok(n) => {
                if let Some(lo) = min {
                    if n < lo {
                        eprintln!("  Value must be >= {lo}.");
                        continue;
                    }
                }
                if let Some(hi) = max {
                    if n > hi {
                        eprintln!("  Value must be <= {hi}.");
                        continue;
                    }
                }
                return Ok(AnswerValue::Number(n));
            }
            Err(_) => {
                eprintln!("  Invalid number. Try again.");
            }
        }
    }
}

fn prompt_multi_select(
    label: &str,
    options: &[llm_questionnaire::ChoiceOption],
    default: Option<&[String]>,
    reader: &mut impl BufRead,
) -> Result<AnswerValue> {
    eprintln!("{label}");
    for (i, opt) in options.iter().enumerate() {
        let marker = default
            .map(|d| d.iter().any(|v| v == &opt.value))
            .unwrap_or(false);
        let tag = if marker { " (default)" } else { "" };
        eprintln!("  {}: {}{tag}", i + 1, opt.label);
        if let Some(desc) = &opt.description {
            eprintln!("       {desc}");
        }
    }
    let default_hint = default
        .map(|d| format!(" [{}]", d.join(",")))
        .unwrap_or_default();
    eprint!("Select (comma-separated numbers, e.g. 1,3){default_hint}: ");
    io::stderr().flush().ok();

    let mut input = String::new();
    reader
        .read_line(&mut input)
        .map_err(|e| llm_core::FrameworkError::questionnaire(format!("read error: {e}")))?;
    let trimmed = input.trim();

    if trimmed.is_empty() {
        if let Some(d) = default {
            return Ok(AnswerValue::MultiSelect(d.to_vec()));
        }
        return Ok(AnswerValue::MultiSelect(vec![]));
    }

    let mut selected = Vec::new();
    for part in trimmed.split(',') {
        let part = part.trim();
        if let Ok(num) = part.parse::<usize>() {
            if num >= 1 && num <= options.len() {
                selected.push(options[num - 1].value.clone());
            } else {
                eprintln!("  Skipping invalid index: {num}");
            }
        } else if let Some(opt) = options.iter().find(|o| o.value == part) {
            selected.push(opt.value.clone());
        } else {
            eprintln!("  Skipping unknown option: {part}");
        }
    }

    Ok(AnswerValue::MultiSelect(selected))
}
