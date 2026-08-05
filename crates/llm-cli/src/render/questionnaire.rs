use std::fmt;
use std::io::{self, BufRead, IsTerminal, Write};

use inquire::Select;
use llm_core::Result;
use llm_questionnaire::{
    AnswerMap, AnswerValue, Question, QuestionKind, Questionnaire, QuestionnaireRun, SectionId,
};

const SELECT_CONTROLS: &str = "Type a shortcut + Enter · ↑/↓ move · Enter select · Esc cancel";
const MULTI_SELECT_CONTROLS: &str =
    "Type a number + Space · ↑/↓ move · Space toggle · Enter confirm · Esc cancel";

/// Action chosen at a prompt instead of (or before) submitting an answer.
#[derive(Debug, Clone, PartialEq)]
enum PromptAction {
    Answer(AnswerValue),
    /// Continue past an info item.
    Continue,
    Back,
    Quit,
}

/// Menu row for interactive ↑/↓ selection (TTY).
#[derive(Debug, Clone)]
enum MenuItem {
    Choice {
        shortcut: usize,
        label: String,
        value: String,
    },
    Yes,
    No,
    Continue,
    Back,
    Quit,
}

impl fmt::Display for MenuItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Choice {
                shortcut, label, ..
            } => write!(f, "{shortcut}  {label}"),
            Self::Yes => write!(f, "1  Yes"),
            Self::No => write!(f, "2  No"),
            Self::Continue => write!(f, "1  Continue"),
            Self::Back => write!(f, "B  ← Back"),
            Self::Quit => write!(f, "Q  Cancel"),
        }
    }
}

/// Drive a [`Questionnaire`] interactively on the terminal, returning the
/// collected answers.
///
/// On a TTY, choice-like prompts support both numbered shortcuts and ↑/↓ +
/// Enter. **← Back** and **Cancel** appear as selectable rows. Going back
/// keeps all answers so earlier work is still there when the user returns
/// forward.
///
/// Non-TTY / scripted input falls back to line prompts (`/back`, `/quit`).
pub fn run_terminal_questionnaire(questionnaire: &Questionnaire) -> Result<AnswerMap> {
    let mut run = QuestionnaireRun::new(questionnaire.clone())
        .map_err(|errs| llm_core::FrameworkError::questionnaire(errs.join("; ")))?;

    let interactive = io::stdin().is_terminal() && io::stderr().is_terminal();
    let stdin = io::stdin();
    let mut reader = stdin.lock();

    print_welcome(questionnaire);

    // Track the last section we printed a header for.
    let mut last_section_id: Option<SectionId> = None;

    while let Some(question) = run.next_question().cloned() {
        let prompt_label = question_prompt_label(&run, &question);

        // Print section header if we've entered a new section.
        if let Some(section) = run.current_section() {
            let sid = &section.id;
            let is_new_section = last_section_id.as_ref() != Some(sid);
            if is_new_section && !section.title.is_empty() {
                eprintln!();
                eprintln!("── {} ──", section.title);
                if !section.description.is_empty() {
                    eprintln!("{}", section.description);
                }
                eprintln!();
                last_section_id = Some(sid.clone());
            } else if is_new_section {
                last_section_id = Some(sid.clone());
            }
        }

        // Prefer a previously recorded answer over the schema default.
        let existing = run.current_answer().cloned();

        // Handle info items: display and wait for acknowledgment.
        if let QuestionKind::Info { content } = &question.kind {
            if !prompt_label.is_empty() {
                eprintln!("{prompt_label}");
            }
            for line in content.lines() {
                eprintln!("    {line}");
            }
            eprintln!();
            match prompt_continue(run.can_go_back(), interactive, &mut reader)? {
                PromptAction::Continue => {
                    run.advance_info()
                        .map_err(|errs| llm_core::FrameworkError::questionnaire(errs.join("; ")))?;
                }
                PromptAction::Back => {
                    if let Err(msg) = apply_go_back(&mut run, &mut last_section_id) {
                        eprintln!("  {msg}");
                    }
                }
                PromptAction::Quit => return abort_questionnaire(),
                PromptAction::Answer(_) => unreachable!("info prompt never answers"),
            }
            continue;
        }

        let action = match &question.kind {
            QuestionKind::Choice { options, default } => {
                let effective_default = existing
                    .as_ref()
                    .and_then(AnswerValue::as_choice)
                    .map(str::to_owned)
                    .or_else(|| default.clone());
                prompt_choice(
                    &prompt_label,
                    question.help_text.as_deref(),
                    options,
                    effective_default.as_deref(),
                    run.can_go_back(),
                    interactive,
                    &mut reader,
                )?
            }
            QuestionKind::YesNo { default } => {
                let effective_default = existing
                    .as_ref()
                    .and_then(AnswerValue::as_yes_no)
                    .or(*default);
                prompt_yes_no(
                    &prompt_label,
                    question.help_text.as_deref(),
                    effective_default,
                    run.can_go_back(),
                    interactive,
                    &mut reader,
                )?
            }
            QuestionKind::Text {
                placeholder,
                default,
            } => {
                let effective_default = existing
                    .as_ref()
                    .and_then(|v| match v {
                        AnswerValue::Text(Some(s)) => Some(s.clone()),
                        _ => None,
                    })
                    .or_else(|| default.clone());
                prompt_text(
                    &prompt_label,
                    question.help_text.as_deref(),
                    placeholder.as_deref(),
                    effective_default.as_deref(),
                    run.can_go_back(),
                    &mut reader,
                )?
            }
            QuestionKind::Number { min, max, default } => {
                let effective_default = existing
                    .as_ref()
                    .and_then(AnswerValue::as_number)
                    .or(*default);
                prompt_number(
                    &prompt_label,
                    question.help_text.as_deref(),
                    *min,
                    *max,
                    effective_default,
                    run.can_go_back(),
                    &mut reader,
                )?
            }
            QuestionKind::MultiSelect { options, default } => {
                let effective_default = existing
                    .as_ref()
                    .and_then(AnswerValue::as_multi_select)
                    .map(|s| s.to_vec())
                    .or_else(|| default.clone());
                prompt_multi_select(
                    &prompt_label,
                    question.help_text.as_deref(),
                    options,
                    effective_default.as_deref(),
                    run.can_go_back(),
                    interactive,
                    &mut reader,
                )?
            }
            QuestionKind::Info { .. } => unreachable!("handled above"),
        };

        match action {
            PromptAction::Answer(answer) => match run.submit_answer(answer) {
                Ok(()) => {}
                Err(errors) => {
                    for err in &errors {
                        eprintln!("  Validation error: {err}");
                    }
                }
            },
            PromptAction::Back => {
                if let Err(msg) = apply_go_back(&mut run, &mut last_section_id) {
                    eprintln!("  {msg}");
                }
            }
            PromptAction::Quit => return abort_questionnaire(),
            PromptAction::Continue => unreachable!("non-info prompts never continue"),
        }
    }

    Ok(run.answers().clone())
}

fn print_welcome(questionnaire: &Questionnaire) {
    if !questionnaire.title.is_empty() {
        eprintln!("{}", questionnaire.title);
    }
    if !questionnaire.description.is_empty() {
        eprintln!("{}", questionnaire.description);
    }
    eprintln!();
}

fn question_prompt_label(run: &QuestionnaireRun, question: &Question) -> String {
    let visible: Vec<&Question> = run
        .questionnaire()
        .questions
        .iter()
        .filter(|candidate| {
            candidate
                .condition
                .as_ref()
                .is_none_or(|condition| condition.evaluate(run.answers()))
        })
        .collect();
    let current = visible
        .iter()
        .position(|candidate| candidate.id == question.id)
        .map(|index| index + 1)
        .unwrap_or(1);
    let total = visible.len().max(current);

    if question.label.is_empty() {
        format!("Question {current} of {total}")
    } else {
        format!("[{current}/{total}] {}", question.label)
    }
}

fn nav_hint(can_go_back: bool) -> &'static str {
    if can_go_back {
        " (/back · /quit)"
    } else {
        " (/quit)"
    }
}

fn abort_questionnaire() -> Result<AnswerMap> {
    eprintln!();
    eprintln!("Questionnaire cancelled. No answers were saved.");
    Err(llm_core::FrameworkError::questionnaire("cancelled by user"))
}

fn apply_go_back(
    run: &mut QuestionnaireRun,
    last_section_id: &mut Option<SectionId>,
) -> std::result::Result<(), String> {
    match run.go_back() {
        Ok(()) => {
            // Force the section header to reprint if we crossed a boundary.
            *last_section_id = None;
            eprintln!("  ← previous question (answers kept)");
            Ok(())
        }
        Err(errors) => Err(errors.join("; ")),
    }
}

fn parse_nav_command(trimmed: &str) -> Option<PromptAction> {
    match trimmed.to_ascii_lowercase().as_str() {
        "/back" | "back" => Some(PromptAction::Back),
        "/quit" | "/exit" | "quit" | "exit" => Some(PromptAction::Quit),
        _ => None,
    }
}

fn parse_nav_command_strict(trimmed: &str) -> Option<PromptAction> {
    match trimmed.to_ascii_lowercase().as_str() {
        "/back" => Some(PromptAction::Back),
        "/quit" | "/exit" => Some(PromptAction::Quit),
        _ => None,
    }
}

fn read_line(reader: &mut impl BufRead) -> Result<Option<String>> {
    let mut input = String::new();
    let n = reader
        .read_line(&mut input)
        .map_err(|e| llm_core::FrameworkError::questionnaire(format!("read error: {e}")))?;
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(input))
}

fn map_inquire_error(err: inquire::InquireError) -> Result<PromptAction> {
    match err {
        inquire::InquireError::OperationCanceled | inquire::InquireError::OperationInterrupted => {
            Ok(PromptAction::Quit)
        }
        other => Err(llm_core::FrameworkError::questionnaire(other.to_string())),
    }
}

fn run_select(
    prompt: &str,
    help_text: Option<&str>,
    items: Vec<MenuItem>,
    starting: usize,
) -> Result<PromptAction> {
    let starting = starting.min(items.len().saturating_sub(1));
    let help = prompt_help(help_text, SELECT_CONTROLS);
    match Select::new(prompt, items)
        .with_starting_cursor(starting)
        .with_help_message(&help)
        .prompt()
    {
        Ok(MenuItem::Choice { value, .. }) => Ok(PromptAction::Answer(AnswerValue::Choice(value))),
        Ok(MenuItem::Yes) => Ok(PromptAction::Answer(AnswerValue::YesNo(true))),
        Ok(MenuItem::No) => Ok(PromptAction::Answer(AnswerValue::YesNo(false))),
        Ok(MenuItem::Continue) => Ok(PromptAction::Continue),
        Ok(MenuItem::Back) => Ok(PromptAction::Back),
        Ok(MenuItem::Quit) => Ok(PromptAction::Quit),
        Err(err) => map_inquire_error(err),
    }
}

fn prompt_help(help_text: Option<&str>, controls: &str) -> String {
    match help_text {
        Some(help) if !help.is_empty() => format!("{help} · {controls}"),
        _ => controls.to_owned(),
    }
}

fn append_nav_items(items: &mut Vec<MenuItem>, can_go_back: bool) {
    if can_go_back {
        items.push(MenuItem::Back);
    }
    items.push(MenuItem::Quit);
}

// ---------------------------------------------------------------------------
// Per-kind prompt helpers
// ---------------------------------------------------------------------------

fn prompt_continue(
    can_go_back: bool,
    interactive: bool,
    reader: &mut impl BufRead,
) -> Result<PromptAction> {
    if interactive {
        let mut items = vec![MenuItem::Continue];
        append_nav_items(&mut items, can_go_back);
        return run_select("Next", None, items, 0);
    }

    loop {
        eprint!("Press Enter to continue{}: ", nav_hint(can_go_back));
        io::stderr().flush().ok();

        let Some(input) = read_line(reader)? else {
            return Ok(PromptAction::Quit);
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(PromptAction::Continue);
        }
        if let Some(action) = parse_nav_command(trimmed) {
            return Ok(action);
        }
        eprintln!("  Press Enter to continue, or use /back / /quit.");
    }
}

fn prompt_choice(
    label: &str,
    help_text: Option<&str>,
    options: &[llm_questionnaire::ChoiceOption],
    default: Option<&str>,
    can_go_back: bool,
    interactive: bool,
    reader: &mut impl BufRead,
) -> Result<PromptAction> {
    if interactive {
        let mut items: Vec<MenuItem> = options
            .iter()
            .enumerate()
            .map(|(index, opt)| {
                let label = match &opt.description {
                    Some(desc) => format!("{} — {desc}", opt.label),
                    None => opt.label.clone(),
                };
                MenuItem::Choice {
                    shortcut: index + 1,
                    label,
                    value: opt.value.clone(),
                }
            })
            .collect();
        let starting = default
            .and_then(|d| options.iter().position(|o| o.value == d))
            .unwrap_or(0);
        append_nav_items(&mut items, can_go_back);
        return run_select(label, help_text, items, starting);
    }

    loop {
        eprintln!("{label}");
        if let Some(help) = help_text {
            eprintln!("  {help}");
        }
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
        if can_go_back {
            eprintln!("  b: ← Back");
        }
        eprintln!("  q: Cancel");

        let default_hint = default.map(|d| format!(" [{d}]")).unwrap_or_default();
        eprint!("Choice{default_hint}{}: ", nav_hint(can_go_back));
        io::stderr().flush().ok();

        let Some(input) = read_line(reader)? else {
            return Ok(PromptAction::Quit);
        };
        let trimmed = input.trim();

        if let Some(action) = parse_nav_command(trimmed) {
            return Ok(action);
        }

        if trimmed.is_empty() {
            if let Some(d) = default {
                return Ok(PromptAction::Answer(AnswerValue::Choice(d.to_owned())));
            }
            eprintln!("  Please enter a selection.");
            continue;
        }

        if let Ok(num) = trimmed.parse::<usize>() {
            if num >= 1 && num <= options.len() {
                return Ok(PromptAction::Answer(AnswerValue::Choice(
                    options[num - 1].value.clone(),
                )));
            }
        }

        if let Some(opt) = options.iter().find(|o| o.value == trimmed) {
            return Ok(PromptAction::Answer(AnswerValue::Choice(opt.value.clone())));
        }

        eprintln!("  Invalid selection. Try again.");
    }
}

fn prompt_yes_no(
    label: &str,
    help_text: Option<&str>,
    default: Option<bool>,
    can_go_back: bool,
    interactive: bool,
    reader: &mut impl BufRead,
) -> Result<PromptAction> {
    if interactive {
        let mut items = vec![MenuItem::Yes, MenuItem::No];
        let starting = match default {
            Some(false) => 1,
            _ => 0,
        };
        append_nav_items(&mut items, can_go_back);
        return run_select(label, help_text, items, starting);
    }

    loop {
        let hint = match default {
            Some(true) => " [Y/n]",
            Some(false) => " [y/N]",
            None => " [y/n]",
        };
        eprintln!("{label}");
        if let Some(help) = help_text {
            eprintln!("  {help}");
        }
        eprint!("Answer{hint}{}: ", nav_hint(can_go_back));
        io::stderr().flush().ok();

        let Some(input) = read_line(reader)? else {
            return Ok(PromptAction::Quit);
        };
        let trimmed = input.trim();

        if let Some(action) = parse_nav_command(trimmed) {
            return Ok(action);
        }

        let lowered = trimmed.to_lowercase();
        if lowered.is_empty() {
            if let Some(d) = default {
                return Ok(PromptAction::Answer(AnswerValue::YesNo(d)));
            }
            eprintln!("  Please enter y or n.");
            continue;
        }

        match lowered.as_str() {
            "y" | "yes" => return Ok(PromptAction::Answer(AnswerValue::YesNo(true))),
            "n" | "no" => return Ok(PromptAction::Answer(AnswerValue::YesNo(false))),
            _ => {
                eprintln!("  Please enter y or n, or use /back / /quit.");
            }
        }
    }
}

fn prompt_text(
    label: &str,
    help_text: Option<&str>,
    placeholder: Option<&str>,
    default: Option<&str>,
    can_go_back: bool,
    reader: &mut impl BufRead,
) -> Result<PromptAction> {
    let hint = placeholder.map(|p| format!(" ({p})")).unwrap_or_default();
    let default_hint = default.map(|d| format!(" [{d}]")).unwrap_or_default();
    eprintln!("{label}{hint}{default_hint}");
    if let Some(help) = help_text {
        eprintln!("  {help}");
    }
    eprint!("Answer{}: ", nav_hint(can_go_back));
    io::stderr().flush().ok();

    let Some(input) = read_line(reader)? else {
        return Ok(PromptAction::Quit);
    };
    let trimmed = input.trim();

    if let Some(action) = parse_nav_command_strict(trimmed) {
        return Ok(action);
    }

    if trimmed.is_empty() {
        if let Some(d) = default {
            Ok(PromptAction::Answer(AnswerValue::Text(Some(d.to_owned()))))
        } else {
            Ok(PromptAction::Answer(AnswerValue::Text(None)))
        }
    } else {
        Ok(PromptAction::Answer(AnswerValue::Text(Some(
            trimmed.to_owned(),
        ))))
    }
}

fn prompt_number(
    label: &str,
    help_text: Option<&str>,
    min: Option<f64>,
    max: Option<f64>,
    default: Option<f64>,
    can_go_back: bool,
    reader: &mut impl BufRead,
) -> Result<PromptAction> {
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
        eprintln!("{label}{range_hint}{default_hint}");
        if let Some(help) = help_text {
            eprintln!("  {help}");
        }
        eprint!("Answer{}: ", nav_hint(can_go_back));
        io::stderr().flush().ok();

        let Some(input) = read_line(reader)? else {
            return Ok(PromptAction::Quit);
        };
        let trimmed = input.trim();

        if let Some(action) = parse_nav_command(trimmed) {
            return Ok(action);
        }

        if trimmed.is_empty() {
            if let Some(d) = default {
                return Ok(PromptAction::Answer(AnswerValue::Number(d)));
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
                return Ok(PromptAction::Answer(AnswerValue::Number(n)));
            }
            Err(_) => {
                eprintln!("  Invalid number. Try again, or use /back / /quit.");
            }
        }
    }
}

fn prompt_multi_select(
    label: &str,
    help_text: Option<&str>,
    options: &[llm_questionnaire::ChoiceOption],
    default: Option<&[String]>,
    can_go_back: bool,
    interactive: bool,
    reader: &mut impl BufRead,
) -> Result<PromptAction> {
    if interactive {
        use inquire::MultiSelect;

        let labels: Vec<String> = options
            .iter()
            .enumerate()
            .map(|(index, o)| match &o.description {
                Some(desc) => format!("{}  {} — {desc}", index + 1, o.label),
                None => format!("{}  {}", index + 1, o.label),
            })
            .collect();

        let defaults: Vec<usize> = default
            .map(|selected| {
                options
                    .iter()
                    .enumerate()
                    .filter_map(|(i, o)| {
                        if selected.iter().any(|v| v == &o.value) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        let help = prompt_help(help_text, MULTI_SELECT_CONTROLS);
        let selected_labels = match MultiSelect::new(label, labels.clone())
            .with_default(&defaults)
            .with_keep_filter(false)
            .with_help_message(&help)
            .prompt()
        {
            Ok(v) => v,
            Err(err) => return map_inquire_error(err),
        };

        let values: Vec<String> = selected_labels
            .iter()
            .filter_map(|picked| {
                labels
                    .iter()
                    .position(|l| l == picked)
                    .map(|i| options[i].value.clone())
            })
            .collect();

        // Offer ← Back as an explicit choice after the selection is confirmed.
        if can_go_back {
            let nav = vec![MenuItem::Continue, MenuItem::Back, MenuItem::Quit];
            match run_select("Continue with this selection?", None, nav, 0)? {
                PromptAction::Continue => {
                    Ok(PromptAction::Answer(AnswerValue::MultiSelect(values)))
                }
                other => Ok(other),
            }
        } else {
            Ok(PromptAction::Answer(AnswerValue::MultiSelect(values)))
        }
    } else {
        eprintln!("{label}");
        if let Some(help) = help_text {
            eprintln!("  {help}");
        }
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
        eprint!(
            "Select (comma-separated numbers, e.g. 1,3){default_hint}{}: ",
            nav_hint(can_go_back)
        );
        io::stderr().flush().ok();

        let Some(input) = read_line(reader)? else {
            return Ok(PromptAction::Quit);
        };
        let trimmed = input.trim();

        if let Some(action) = parse_nav_command(trimmed) {
            return Ok(action);
        }

        if trimmed.is_empty() {
            if let Some(d) = default {
                return Ok(PromptAction::Answer(AnswerValue::MultiSelect(d.to_vec())));
            }
            return Ok(PromptAction::Answer(AnswerValue::MultiSelect(vec![])));
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

        Ok(PromptAction::Answer(AnswerValue::MultiSelect(selected)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm_questionnaire::{ChoiceOption, Question, QuestionId, QuestionnaireId};
    use std::io::Cursor;

    fn sample_questionnaire() -> Questionnaire {
        Questionnaire {
            id: QuestionnaireId::new("nav"),
            title: "Nav Test".into(),
            description: "Exercise back and quit.".into(),
            sections: vec![],
            questions: vec![
                Question {
                    id: QuestionId::new("q1"),
                    label: "First?".into(),
                    help_text: None,
                    kind: QuestionKind::YesNo {
                        default: Some(true),
                    },
                    required: true,
                    validation: vec![],
                    condition: None,
                },
                Question {
                    id: QuestionId::new("q2"),
                    label: "Second?".into(),
                    help_text: None,
                    kind: QuestionKind::Choice {
                        options: vec![
                            ChoiceOption {
                                value: "a".into(),
                                label: "A".into(),
                                description: None,
                            },
                            ChoiceOption {
                                value: "b".into(),
                                label: "B".into(),
                                description: None,
                            },
                        ],
                        default: Some("a".into()),
                    },
                    required: true,
                    validation: vec![],
                    condition: None,
                },
            ],
        }
    }

    #[test]
    fn parse_nav_commands() {
        assert_eq!(parse_nav_command("/back"), Some(PromptAction::Back));
        assert_eq!(parse_nav_command("BACK"), Some(PromptAction::Back));
        assert_eq!(parse_nav_command("/quit"), Some(PromptAction::Quit));
        assert_eq!(parse_nav_command("quit"), Some(PromptAction::Quit));
        assert_eq!(parse_nav_command("yes"), None);
        assert_eq!(parse_nav_command_strict("quit"), None);
        assert_eq!(parse_nav_command_strict("/quit"), Some(PromptAction::Quit));
    }

    #[test]
    fn yes_no_accepts_back_and_quit() {
        let mut reader = Cursor::new("back\n");
        let action = prompt_yes_no("Go?", None, Some(true), true, false, &mut reader).unwrap();
        assert_eq!(action, PromptAction::Back);

        let mut reader = Cursor::new("/quit\n");
        let action = prompt_yes_no("Go?", None, Some(true), true, false, &mut reader).unwrap();
        assert_eq!(action, PromptAction::Quit);
    }

    #[test]
    fn eof_is_quit() {
        let mut reader = Cursor::new("");
        let action = prompt_yes_no("Go?", None, Some(true), false, false, &mut reader).unwrap();
        assert_eq!(action, PromptAction::Quit);
    }

    #[test]
    fn text_allows_bare_quit_as_answer() {
        let mut reader = Cursor::new("quit\n");
        let action = prompt_text("Notes", None, None, None, true, &mut reader).unwrap();
        assert_eq!(
            action,
            PromptAction::Answer(AnswerValue::Text(Some("quit".into())))
        );
    }

    #[test]
    fn go_back_keeps_answers_and_prefills() {
        let mut reader = Cursor::new("y\n");
        let q = sample_questionnaire();
        let mut run = QuestionnaireRun::new(q).unwrap();

        let a1 = prompt_yes_no("First?", None, Some(true), false, false, &mut reader).unwrap();
        match a1 {
            PromptAction::Answer(v) => run.submit_answer(v).unwrap(),
            other => panic!("expected answer, got {other:?}"),
        }
        assert_eq!(run.next_question().unwrap().id.as_str(), "q2");

        let mut reader = Cursor::new("/back\n");
        let options = [
            ChoiceOption {
                value: "a".into(),
                label: "A".into(),
                description: None,
            },
            ChoiceOption {
                value: "b".into(),
                label: "B".into(),
                description: None,
            },
        ];
        let back = prompt_choice(
            "Second?",
            None,
            &options,
            Some("a"),
            true,
            false,
            &mut reader,
        )
        .unwrap();
        assert_eq!(back, PromptAction::Back);

        run.go_back().unwrap();
        assert_eq!(run.next_question().unwrap().id.as_str(), "q1");
        assert_eq!(run.current_answer(), Some(&AnswerValue::YesNo(true)));
        // Later work is still present even after going back.
        // (q2 was never answered in this scenario — answer q2 first.)
    }

    #[test]
    fn go_back_to_beginning_keeps_all_answers() {
        let q = sample_questionnaire();
        let mut run = QuestionnaireRun::new(q).unwrap();
        run.submit_answer(AnswerValue::YesNo(true)).unwrap();
        run.submit_answer(AnswerValue::Choice("b".into())).unwrap();
        assert!(run.is_complete());

        run.go_back().unwrap();
        run.go_back().unwrap();
        assert_eq!(run.next_question().unwrap().id.as_str(), "q1");
        assert_eq!(run.answers().len(), 2);
        assert_eq!(run.answers().choice(&QuestionId::new("q2")), Some("b"));
    }

    #[test]
    fn interactive_rows_expose_filterable_shortcuts() {
        let item = MenuItem::Choice {
            shortcut: 2,
            label: "Anthropic".into(),
            value: "anthropic".into(),
        };
        let rendered = item.to_string();

        assert_eq!(rendered, "2  Anthropic");
        assert!(Select::<MenuItem>::DEFAULT_SCORER("2", &item, &rendered, 0).is_some());
        assert_eq!(MenuItem::Back.to_string(), "B  ← Back");
        assert_eq!(MenuItem::Quit.to_string(), "Q  Cancel");
    }

    #[test]
    fn prompt_label_tracks_progress() {
        let q = sample_questionnaire();
        let mut run = QuestionnaireRun::new(q).unwrap();
        let first = run.next_question().unwrap();
        assert_eq!(question_prompt_label(&run, first), "[1/2] First?");

        run.submit_answer(AnswerValue::YesNo(true)).unwrap();
        let second = run.next_question().unwrap();
        assert_eq!(question_prompt_label(&run, second), "[2/2] Second?");
    }
}
