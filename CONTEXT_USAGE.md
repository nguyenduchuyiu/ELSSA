# ELSSA Context Management System

## Overview

ELSSA now includes a comprehensive context management system that automatically saves and manages conversation history. Each conversation session is stored with timestamps and can be viewed, searched, and managed through various tools.

## Features

- ✅ **Automatic Context Saving**: All conversations are automatically saved to JSON files
- ✅ **Session Management**: Each conversation session gets a unique ID with start/end times
- ✅ **Context Trimming**: Keeps conversation context manageable (configurable length)
- ✅ **Search & Export**: Find and export specific conversations
- ✅ **Cleanup Tools**: Remove old conversations automatically

## File Structure

```
data/
└── context/
    ├── session_20241201_143022.json
    ├── session_20241201_150315.json
    └── session_20241201_162144.json
```

## Configuration

In your `ELSSAConfig`, you can configure:

```python
self.context_dir = "data/context"        # Where to store conversations
self.max_context_length = 10             # Max messages in context window
```

## Context Viewer Tool

Use the `context_viewer.py` script to manage conversations:

### List Recent Sessions
```bash
python context_viewer.py list
python context_viewer.py list --limit 20
```

### View Specific Session
```bash
python context_viewer.py show 20241201_143022
python context_viewer.py show 20241201_143022 --format json
```

### Search Conversations
```bash
python context_viewer.py search "weather"
python context_viewer.py search "how are you"
```

### Export Session
```bash
python context_viewer.py export 20241201_143022 conversation.txt
python context_viewer.py export 20241201_143022 conversation.json
```

### Delete Session
```bash
python context_viewer.py delete 20241201_143022
```

### Cleanup Old Sessions
```bash
python context_viewer.py cleanup --days 30
```

## Session Format

Each conversation session is stored as JSON:

```json
{
  "session_id": "20241201_143022",
  "start_time": "2024-12-01T14:30:22.123456",
  "end_time": "2024-12-01T14:45:18.654321",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant named ELSSA.",
      "timestamp": "2024-12-01T14:30:22.123456",
      "session_id": "20241201_143022"
    },
    {
      "role": "user",
      "content": "Hello, how are you?",
      "timestamp": "2024-12-01T14:30:25.789012",
      "session_id": "20241201_143022"
    },
    {
      "role": "assistant",
      "content": "Hello! I'm ELSSA, and I'm doing well. How can I help you today?",
      "timestamp": "2024-12-01T14:30:28.345678",
      "session_id": "20241201_143022"
    }
  ]
}
```

## Integration in Main System

The context management is now fully integrated into the main ELSSA system:

1. **Session Start**: When transitioning to ACTIVE state, a new session begins
2. **Message Logging**: User inputs and assistant responses are automatically saved
3. **Context Retrieval**: The LLM receives the full conversation context
4. **Session End**: When returning to IDLE state, the session is marked as ended

## Key Classes

### `ChatMessage`
Represents a single message with role, content, timestamp, and session ID.

### `ConversationSession`
Contains all messages for a conversation session with metadata.

### `ContextManager`
Main class that handles:
- Session creation and management
- Message storage and retrieval
- Context trimming for LLM
- File I/O operations
- Search and cleanup functions

## Benefits

1. **Conversation Continuity**: ELSSA remembers the context within each session
2. **Better Responses**: LLM has access to conversation history for more relevant responses
3. **Data Analytics**: Analyze conversation patterns and improve the system
4. **User Experience**: More natural conversations with memory
5. **Debugging**: Easy to review and debug conversation flows

## Example Usage

```python
# Initialize context manager
context_manager = ContextManager(
    context_dir="data/context",
    max_context_length=10
)

# Start a new session
session_id = await context_manager.start_new_session()

# Add messages
await context_manager.add_message("user", "Hello!")
await context_manager.add_message("assistant", "Hi there!")

# Get context for LLM
context = await context_manager.get_conversation_context()

# End session
await context_manager.end_current_session()
```

## Notes

- Sessions are automatically saved after each message
- Context is trimmed to prevent token limits (keeps system message + recent messages)
- Files are stored in UTF-8 encoding to support multiple languages
- Session IDs use timestamp format: `YYYYMMDD_HHMMSS` 