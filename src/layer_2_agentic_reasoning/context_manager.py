import json
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: str
    session_id: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        return cls(**data)


@dataclass
class ConversationSession:
    """Represents a conversation session"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    messages: List[ChatMessage] = None
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "messages": [msg.to_dict() for msg in self.messages]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationSession':
        messages = [ChatMessage.from_dict(msg) for msg in data.get("messages", [])]
        return cls(
            session_id=data["session_id"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            messages=messages
        )


class ContextManager:
    """Manages conversation context and history"""
    
    def __init__(self, context_dir: str = "data/context", max_context_length: int = 10):
        self.context_dir = Path(context_dir)
        self.max_context_length = max_context_length
        self.current_session: Optional[ConversationSession] = None
        self._lock = asyncio.Lock()
        
        # Create context directory if it doesn't exist
        self.context_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_session_file_path(self, session_id: str) -> Path:
        """Get the file path for a session"""
        return self.context_dir / f"session_{session_id}.json"
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.now().isoformat()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async def start_new_session(self) -> str:
        """Start a new conversation session"""
        async with self._lock:
            # End current session if exists
            if self.current_session:
                await self.end_current_session()
            
            session_id = self._generate_session_id()
            self.current_session = ConversationSession(
                session_id=session_id,
                start_time=self._get_current_timestamp()
            )
            
            # Add system message
            system_msg = ChatMessage(
                role="system",
                content="You are a helpful assistant named ELSSA.",
                timestamp=self._get_current_timestamp(),
                session_id=session_id
            )
            self.current_session.messages.append(system_msg)
            
            print(f"🆕 Started new conversation session: {session_id}")
            return session_id
    
    async def add_message(self, role: str, content: str) -> None:
        """Add a message to current session"""
        async with self._lock:
            if not self.current_session:
                await self.start_new_session()
            
            message = ChatMessage(
                role=role,
                content=content,
                timestamp=self._get_current_timestamp(),
                session_id=self.current_session.session_id
            )
            
            self.current_session.messages.append(message)
            
            # Trim context if too long (keep system message + recent messages)
            if len(self.current_session.messages) > self.max_context_length + 1:
                # Keep system message (first) + recent messages
                system_msg = self.current_session.messages[0]
                recent_messages = self.current_session.messages[-(self.max_context_length):]
                self.current_session.messages = [system_msg] + recent_messages
                print(f"🔄 Trimmed context to {len(self.current_session.messages)} messages")
            
            # Auto-save after each message
            await self._save_current_session()
    
    async def get_conversation_context(self) -> List[Dict]:
        """Get current conversation context for LLM"""
        async with self._lock:
            if not self.current_session:
                await self.start_new_session()
            
            # Return messages in LLM format
            return [
                {"role": msg.role, "content": msg.content} 
                for msg in self.current_session.messages
            ]
    
    async def end_current_session(self) -> None:
        """End the current conversation session"""
        async with self._lock:
            if self.current_session:
                self.current_session.end_time = self._get_current_timestamp()
                await self._save_current_session()
                print(f"📝 Ended conversation session: {self.current_session.session_id}")
                self.current_session = None
    
    async def _save_current_session(self) -> None:
        """Save current session to file"""
        if not self.current_session:
            return
        
        session_file = self._get_session_file_path(self.current_session.session_id)
        
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_session.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Error saving session: {e}")
    
    async def load_session(self, session_id: str) -> Optional[ConversationSession]:
        """Load a specific session from file"""
        session_file = self._get_session_file_path(session_id)
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ConversationSession.from_dict(data)
        except Exception as e:
            print(f"⚠️ Error loading session {session_id}: {e}")
            return None
    
    async def get_recent_sessions(self, limit: int = 10) -> List[str]:
        """Get list of recent session IDs"""
        session_files = list(self.context_dir.glob("session_*.json"))
        session_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return [f.stem.replace("session_", "") for f in session_files[:limit]]
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a specific session"""
        session_file = self._get_session_file_path(session_id)
        
        try:
            if session_file.exists():
                session_file.unlink()
                print(f"🗑️ Deleted session: {session_id}")
                return True
            return False
        except Exception as e:
            print(f"⚠️ Error deleting session {session_id}: {e}")
            return False
    
    async def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get summary of a session"""
        session = await self.load_session(session_id)
        if not session:
            return None
        
        user_messages = [msg for msg in session.messages if msg.role == "user"]
        assistant_messages = [msg for msg in session.messages if msg.role == "assistant"]
        
        return {
            "session_id": session_id,
            "start_time": session.start_time,
            "end_time": session.end_time,
            "total_messages": len(session.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "duration": self._calculate_duration(session.start_time, session.end_time)
        }
    
    def _calculate_duration(self, start_time: str, end_time: Optional[str]) -> Optional[str]:
        """Calculate session duration"""
        if not end_time:
            return None
        
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            return str(duration)
        except Exception:
            return None
    
    async def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days"""
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        session_files = list(self.context_dir.glob("session_*.json"))
        deleted_count = 0
        
        for session_file in session_files:
            if session_file.stat().st_mtime < cutoff_time:
                try:
                    session_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"⚠️ Error deleting old session {session_file.name}: {e}")
        
        if deleted_count > 0:
            print(f"🧹 Cleaned up {deleted_count} old sessions")
        
        return deleted_count 