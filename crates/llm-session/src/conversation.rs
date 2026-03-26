use llm_core::{ContentBlock, Message, Role};

/// The canonical conversation transcript.
///
/// `ConversationState` owns the ordered list of [`Message`]s exchanged between
/// the user, assistant, and tools.  It provides convenience methods for
/// appending common message shapes while keeping the underlying vector private.
#[derive(Debug, Clone, Default)]
pub struct ConversationState {
    messages: Vec<Message>,
}

impl ConversationState {
    /// Create an empty conversation.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a user text message.
    pub fn append_user(&mut self, text: impl Into<String>) {
        self.messages.push(Message::user(text));
    }

    /// Append an assistant text message.
    pub fn append_assistant(&mut self, text: impl Into<String>) {
        self.messages.push(Message::assistant(text));
    }

    /// Append an assistant message that contains one or more tool-use blocks.
    ///
    /// This is the canonical way to record the assistant requesting tool calls.
    pub fn append_tool_use(&mut self, blocks: Vec<ContentBlock>) {
        self.messages.push(Message {
            role: Role::Assistant,
            content: blocks,
            metadata: Default::default(),
        });
    }

    /// Append a tool result message.
    pub fn append_tool_result(
        &mut self,
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
    ) {
        self.messages
            .push(Message::tool_result(tool_use_id, content));
    }

    /// Append a raw [`Message`] directly.
    pub fn append_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Borrow the full message history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Return the number of messages in the conversation.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns `true` if the conversation contains no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_and_retrieve() {
        let mut conv = ConversationState::new();
        assert!(conv.is_empty());

        conv.append_user("Hello");
        conv.append_assistant("Hi there!");
        conv.append_tool_result("call-1", r#"{"result": "ok"}"#);

        assert_eq!(conv.len(), 3);
        assert!(!conv.is_empty());

        let msgs = conv.messages();
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[0].text_content(), "Hello");
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[1].text_content(), "Hi there!");
        assert_eq!(msgs[2].role, Role::Tool);
    }

    #[test]
    fn append_tool_use_blocks() {
        let mut conv = ConversationState::new();
        conv.append_tool_use(vec![
            ContentBlock::Text("Let me call a tool.".into()),
            ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "echo".into(),
                input: serde_json::json!({"message": "hi"}),
            },
        ]);

        assert_eq!(conv.len(), 1);
        assert_eq!(conv.messages()[0].role, Role::Assistant);
        assert_eq!(conv.messages()[0].content.len(), 2);
    }

    #[test]
    fn append_raw_message() {
        let mut conv = ConversationState::new();
        conv.append_message(Message::system("You are a helpful assistant."));
        assert_eq!(conv.len(), 1);
        assert_eq!(conv.messages()[0].role, Role::System);
    }
}
