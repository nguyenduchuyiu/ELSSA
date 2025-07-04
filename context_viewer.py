#!/usr/bin/env python3
"""
Context Viewer - Utility to view and manage ELSSA conversation history
"""

import argparse
import asyncio
import json
from pathlib import Path
from src.layer_2_agentic_reasoning.context_manager import ContextManager


async def list_sessions(context_manager: ContextManager, limit: int = 10):
    """List recent conversation sessions"""
    sessions = await context_manager.get_recent_sessions(limit)
    
    if not sessions:
        print("📭 No conversation sessions found.")
        return
    
    print(f"📚 Recent conversation sessions (limit: {limit}):")
    print("-" * 60)
    
    for session_id in sessions:
        summary = await context_manager.get_session_summary(session_id)
        if summary:
            print(f"🆔 Session: {session_id}")
            print(f"⏰ Start: {summary['start_time']}")
            if summary['end_time']:
                print(f"⏹️  End: {summary['end_time']}")
                if summary['duration']:
                    print(f"⌛ Duration: {summary['duration']}")
            else:
                print("🔴 Status: Active")
            print(f"💬 Messages: {summary['total_messages']} total, {summary['user_messages']} user, {summary['assistant_messages']} assistant")
            print("-" * 60)


async def show_session(context_manager: ContextManager, session_id: str, format_type: str = "pretty"):
    """Show detailed conversation for a specific session"""
    session = await context_manager.load_session(session_id)
    
    if not session:
        print(f"❌ Session '{session_id}' not found.")
        return
    
    if format_type == "json":
        print(json.dumps(session.to_dict(), indent=2, ensure_ascii=False))
        return
    
    # Pretty format
    print(f"📋 Conversation Session: {session_id}")
    print(f"⏰ Started: {session.start_time}")
    if session.end_time:
        print(f"⏹️  Ended: {session.end_time}")
    print("=" * 80)
    
    for i, message in enumerate(session.messages, 1):
        role_icon = {
            "system": "🔧",
            "user": "👤",
            "assistant": "🤖"
        }.get(message.role, "❓")
        
        print(f"\n{role_icon} {message.role.upper()} [{message.timestamp}]")
        print("-" * 40)
        print(message.content)
        
        if i < len(session.messages):
            print()


async def delete_session(context_manager: ContextManager, session_id: str):
    """Delete a specific session"""
    success = await context_manager.delete_session(session_id)
    
    if success:
        print(f"✅ Session '{session_id}' deleted successfully.")
    else:
        print(f"❌ Failed to delete session '{session_id}' or session not found.")


async def cleanup_old_sessions(context_manager: ContextManager, days: int):
    """Clean up old sessions"""
    deleted_count = await context_manager.cleanup_old_sessions(days)
    print(f"🧹 Cleaned up {deleted_count} sessions older than {days} days.")


async def search_sessions(context_manager: ContextManager, query: str):
    """Search for sessions containing specific text"""
    sessions = await context_manager.get_recent_sessions(50)  # Search more sessions
    matching_sessions = []
    
    for session_id in sessions:
        session = await context_manager.load_session(session_id)
        if session:
            # Search in message content
            for message in session.messages:
                if query.lower() in message.content.lower():
                    matching_sessions.append((session_id, message))
                    break
    
    if not matching_sessions:
        print(f"🔍 No sessions found containing '{query}'")
        return
    
    print(f"🔍 Found {len(matching_sessions)} sessions containing '{query}':")
    print("-" * 60)
    
    for session_id, message in matching_sessions:
        summary = await context_manager.get_session_summary(session_id)
        print(f"🆔 Session: {session_id}")
        print(f"⏰ Start: {summary['start_time']}")
        print(f"📝 Sample: {message.content[:100]}{'...' if len(message.content) > 100 else ''}")
        print("-" * 60)


async def export_session(context_manager: ContextManager, session_id: str, output_file: str):
    """Export session to a file"""
    session = await context_manager.load_session(session_id)
    
    if not session:
        print(f"❌ Session '{session_id}' not found.")
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            if output_file.endswith('.json'):
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
            else:
                # Export as text
                f.write(f"ELSSA Conversation Session: {session_id}\n")
                f.write(f"Started: {session.start_time}\n")
                if session.end_time:
                    f.write(f"Ended: {session.end_time}\n")
                f.write("=" * 80 + "\n\n")
                
                for message in session.messages:
                    f.write(f"{message.role.upper()} [{message.timestamp}]:\n")
                    f.write(f"{message.content}\n\n")
        
        print(f"✅ Session exported to '{output_file}'")
    except Exception as e:
        print(f"❌ Error exporting session: {e}")


async def main():
    parser = argparse.ArgumentParser(description="ELSSA Context Viewer - Manage conversation history")
    parser.add_argument("--context-dir", default="data/context", help="Context directory path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List sessions
    list_parser = subparsers.add_parser("list", help="List recent sessions")
    list_parser.add_argument("--limit", type=int, default=10, help="Number of sessions to show")
    
    # Show session
    show_parser = subparsers.add_parser("show", help="Show specific session")
    show_parser.add_argument("session_id", help="Session ID to show")
    show_parser.add_argument("--format", choices=["pretty", "json"], default="pretty", help="Output format")
    
    # Delete session
    delete_parser = subparsers.add_parser("delete", help="Delete specific session")
    delete_parser.add_argument("session_id", help="Session ID to delete")
    
    # Cleanup old sessions
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old sessions")
    cleanup_parser.add_argument("--days", type=int, default=30, help="Delete sessions older than N days")
    
    # Search sessions
    search_parser = subparsers.add_parser("search", help="Search sessions")
    search_parser.add_argument("query", help="Text to search for")
    
    # Export session
    export_parser = subparsers.add_parser("export", help="Export session to file")
    export_parser.add_argument("session_id", help="Session ID to export")
    export_parser.add_argument("output_file", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize context manager
    context_manager = ContextManager(context_dir=args.context_dir)
    
    try:
        if args.command == "list":
            await list_sessions(context_manager, args.limit)
        elif args.command == "show":
            await show_session(context_manager, args.session_id, args.format)
        elif args.command == "delete":
            await delete_session(context_manager, args.session_id)
        elif args.command == "cleanup":
            await cleanup_old_sessions(context_manager, args.days)
        elif args.command == "search":
            await search_sessions(context_manager, args.query)
        elif args.command == "export":
            await export_session(context_manager, args.session_id, args.output_file)
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 