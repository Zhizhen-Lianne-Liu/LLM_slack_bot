from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import json
import asyncio
import os
import threading
from flask import Flask, request
from slackeventsapi import SlackEventAdapter
import requests
import slack
from datetime import datetime, timedelta
from collections import defaultdict

load_dotenv('.env')

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

class ConversationHistory:
    def __init__(self, max_messages_per_channel=50, history_timeout_hours=24):
        self.conversations = defaultdict(list)  # channel_id -> list of messages
        self.last_activity = defaultdict(datetime)  # channel_id -> last message time
        self.max_messages = max_messages_per_channel
        self.timeout_hours = history_timeout_hours
    
    def add_message(self, channel_id: str, role: str, content: str):
        """Add a message to the conversation history"""
        now = datetime.now()
        self.last_activity[channel_id] = now
        
        # Clean up old conversations first
        self._cleanup_old_conversations()
        
        # Add the new message
        self.conversations[channel_id].append({
            'role': role,
            'content': content,
            'timestamp': now
        })
        
        # Keep only the most recent messages to prevent token limit issues
        if len(self.conversations[channel_id]) > self.max_messages:
            self.conversations[channel_id] = self.conversations[channel_id][-self.max_messages:]
    
    def get_messages(self, channel_id: str) -> List[Dict]:
        """Get conversation history for a channel in Anthropic format"""
        self._cleanup_old_conversations()
        
        messages = []
        for msg in self.conversations[channel_id]:
            if msg['role'] in ['user', 'assistant']:
                messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        return messages
    
    def clear_channel(self, channel_id: str):
        """Clear conversation history for a specific channel"""
        if channel_id in self.conversations:
            del self.conversations[channel_id]
        if channel_id in self.last_activity:
            del self.last_activity[channel_id]
    
    def _cleanup_old_conversations(self):
        """Remove conversations that haven't been active recently"""
        cutoff_time = datetime.now() - timedelta(hours=self.timeout_hours)
        channels_to_remove = [
            channel_id for channel_id, last_time in self.last_activity.items()
            if last_time < cutoff_time
        ]
        
        for channel_id in channels_to_remove:
            if channel_id in self.conversations:
                del self.conversations[channel_id]
            if channel_id in self.last_activity:
                del self.last_activity[channel_id]

class MCP_SlackBot:
    def __init__(self):
        # Initialize Flask app and Slack components
        self.app = Flask(__name__)
        self.slack_event_adapter = SlackEventAdapter(
            os.environ['SLACK_SIGNING_TOKEN'], '/slack/events', self.app
        )
        
        # Add conversation memory
        self.conversation_history = ConversationHistory()
        
        # Add a basic health check endpoint
        @self.app.route('/')
        def health_check():
            return {'status': 'MCP Slack Bot is running', 'active_channels': len(self.active_channels)}
        
        # Add endpoint to clear conversation history
        @self.app.route('/clear/<channel_id>', methods=['POST'])
        def clear_conversation(channel_id):
            self.conversation_history.clear_channel(channel_id)
            return {'status': f'Cleared conversation history for channel {channel_id}'}
        
        # Add logging for debugging
        @self.app.before_request
        def log_request():
            print(f"Received {request.method} request to {request.path}")
            if request.json:
                print(f"Request body: {request.json}")
        
        self.client = slack.WebClient(os.environ["SLACK_BOT_TOKEN"])
        
        # Test the connection and get bot info
        try:
            auth_test = self.client.api_call("auth.test")
            self.BOT_ID = auth_test['user_id']
            print(f"✅ Bot connected successfully!")
            print(f"Bot ID: {self.BOT_ID}")
            print(f"Bot Name: {auth_test.get('user', 'Unknown')}")
            print(f"Team: {auth_test.get('team', 'Unknown')}")
        except Exception as e:
            print(f"❌ Failed to connect to Slack: {e}")
            raise
        
        # Initialize MCP components
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        
        # Bot state management
        self.is_running = False
        self.mcp_initialization_error = None
        self.active_channels = set()  # Channels where bot is active
        self.loop = None
        
        # Setup event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        @self.slack_event_adapter.on('app_mention')
        def app_mention(payload):
            print(f"🔔 App mention received: {payload}")
            event = payload.get('event', {})
            channel_id = event.get('channel')
            user_id = event.get('user')
            text = event.get('text', '')
            
            print(f"Channel: {channel_id}, User: {user_id}, Text: {text}")
            
            if self.BOT_ID != user_id:
                # Check for special commands
                if 'start' in text.lower() or 'begin' in text.lower():
                    self._start_bot_in_channel(channel_id)
                elif 'quit' in text.lower() or 'stop' in text.lower():
                    self._stop_bot_in_channel(channel_id)
                elif 'clear' in text.lower() and 'history' in text.lower():
                    self._clear_conversation_history(channel_id)
                elif 'reset' in text.lower():
                    self._clear_conversation_history(channel_id)
                elif channel_id in self.active_channels:
                    # Process the query if bot is active in this channel
                    # Remove the bot mention from the text
                    clean_text = self._clean_mention_from_text(text)
                    if clean_text.strip():
                        # Add user message to history
                        self.conversation_history.add_message(channel_id, 'user', clean_text)
                        self._process_slack_query(channel_id, clean_text)

        @self.slack_event_adapter.on('message')
        def message(payload):
            print(f"📨 Message received: {payload}")
            event = payload.get('event', {})
            channel_id = event.get('channel')
            user_id = event.get('user')
            text = event.get('text', '')
            
            print(f"Channel: {channel_id}, User: {user_id}, Text: {text}")
            
            # Only respond to messages in active channels and not from the bot itself
            if (self.BOT_ID != user_id and 
                channel_id in self.active_channels and 
                not event.get('subtype')):  # Ignore message subtypes like bot messages
                
                if 'quit' in text.lower() or 'stop' in text.lower():
                    self._stop_bot_in_channel(channel_id)
                elif 'clear' in text.lower() and 'history' in text.lower():
                    self._clear_conversation_history(channel_id)
                elif 'reset' in text.lower():
                    self._clear_conversation_history(channel_id)
                else:
                    # Add user message to history
                    self.conversation_history.add_message(channel_id, 'user', text)
                    self._process_slack_query(channel_id, text)

        # Add error handler
        @self.slack_event_adapter.on('error')
        def error_handler(err):
            print(f"❌ Slack event error: {err}")

        # Add a catch-all handler to see what events we're getting
        @self.slack_event_adapter.on('*')
        def catch_all(payload):
            event_type = payload.get('event', {}).get('type', 'unknown')
            print(f"🔍 Received event type: {event_type}")
            if event_type not in ['app_mention', 'message']:
                print(f"   Full payload: {payload}")

    def _clean_mention_from_text(self, text: str) -> str:
        """Remove bot mention from text"""
        # Remove <@BOTID> mentions
        import re
        pattern = f'<@{self.BOT_ID}>'
        return re.sub(pattern, '', text).strip()

    def _clear_conversation_history(self, channel_id: str):
        """Clear conversation history for a channel"""
        self.conversation_history.clear_channel(channel_id)
        self.client.chat_postMessage(
            channel=channel_id, 
            text="🧹 Conversation history cleared! Starting fresh."
        )

    def _start_bot_in_channel(self, channel_id: str):
        """Start the bot in a specific channel"""
        print(f"🔄 Starting bot in channel {channel_id}")
        
        if not self.is_running and not self.loop:
            print("🔧 Initializing MCP connections...")
            # Start the MCP connections in a separate thread
            threading.Thread(target=self._initialize_mcp, daemon=True).start()
            
            # Give some time for initialization to start
            import time
            time.sleep(1)
        
        self.active_channels.add(channel_id)
        
        if self.mcp_initialization_error:
            self.client.chat_postMessage(
                channel=channel_id, 
                text=f"❌ Failed to initialize MCP: {self.mcp_initialization_error}.\nBot will work without MCP tools.\n\n💡 Commands:\n• 'quit' or 'stop' - Stop bot\n• 'clear history' or 'reset' - Clear conversation memory"
            )
            self.is_running = True  # Allow basic functionality
        elif self.is_running:
            tools_text = ', '.join([tool['name'] for tool in self.available_tools]) if self.available_tools else 'None'
            self.client.chat_postMessage(
                channel=channel_id, 
                text=f"🤖 MCP Chatbot is now active with conversation memory!\n\n🔧 Available tools: {tools_text}\n\n💡 Commands:\n• 'quit' or 'stop' - Stop bot\n• 'clear history' or 'reset' - Clear conversation memory"
            )
        else:
            self.client.chat_postMessage(
                channel=channel_id, 
                text="🤖 MCP Chatbot is initializing... This should take a few seconds. I'll let you know when ready!"
            )

    def _stop_bot_in_channel(self, channel_id: str):
        """Stop the bot in a specific channel"""
        if channel_id in self.active_channels:
            self.active_channels.remove(channel_id)
            # Optionally clear history when stopping
            # self.conversation_history.clear_channel(channel_id)
            self.client.chat_postMessage(
                channel=channel_id, 
                text="🤖 MCP Chatbot stopped in this channel. Mention me with 'start' to reactivate.\n💾 Conversation history is preserved for 24 hours."
            )

    def _initialize_mcp(self):
        """Initialize MCP connections in a separate event loop"""
        print("🔧 Starting MCP initialization thread...")
        try:
            # Create new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # Run the async initialization
            print("🔧 Running async MCP initialization...")
            self.loop.run_until_complete(self._async_initialize_mcp())
            self.is_running = True
            
            print("✅ MCP initialization complete!")
            
            # Notify all active channels that the bot is ready
            for channel_id in self.active_channels:
                tools_list = ', '.join([tool['name'] for tool in self.available_tools]) if self.available_tools else 'None'
                self.client.chat_postMessage(
                    channel=channel_id, 
                    text=f"✅ MCP Chatbot is ready with conversation memory!\n🔧 Available tools: {tools_list}\n\n💡 Commands:\n• 'quit' or 'stop' - Stop bot\n• 'clear history' or 'reset' - Clear conversation memory"
                )
            
            # Keep the loop running for async operations
            self.loop.run_forever()
        except Exception as e:
            error_msg = f"Error initializing MCP: {e}"
            print(f"❌ {error_msg}")
            self.mcp_initialization_error = str(e)
            
            # Notify active channels about the error
            for channel_id in self.active_channels:
                self.client.chat_postMessage(
                    channel=channel_id, 
                    text=f"❌ MCP initialization failed: {str(e)}\nBot will work without MCP tools but with conversation memory.\n\n💡 Commands:\n• 'quit' or 'stop' - Stop bot\n• 'clear history' or 'reset' - Clear conversation memory"
                )

    async def _async_initialize_mcp(self):
        """Async MCP initialization"""
        print("🔧 Connecting to MCP servers...")
        await self.connect_to_servers()
        print(f"✅ Connected to {len(self.sessions)} MCP servers with {len(self.available_tools)} total tools")

    def _process_slack_query(self, channel_id: str, query: str):
        """Process a query from Slack"""
        print(f"🤔 Processing query: {query}")
        self.client.chat_postMessage(
                    channel=channel_id, 
                    text=f"🤔 Processing query"
                )
        
        if not self.is_running:
            if self.mcp_initialization_error:
                # Process without MCP tools but with conversation history
                self._process_query_without_mcp(channel_id, query)
            else:
                self.client.chat_postMessage(
                    channel=channel_id, 
                    text="🤖 Bot is still initializing... Please wait a moment."
                )
            return
        
        if not self.loop:
            self.client.chat_postMessage(
                channel=channel_id, 
                text="❌ Bot event loop not available. Please restart the bot."
            )
            return
        
        # Schedule the async query processing
        future = asyncio.run_coroutine_threadsafe(
            self.process_slack_query(channel_id, query), 
            self.loop
        )

    def _process_query_without_mcp(self, channel_id: str, query: str):
        """Process query without MCP tools (fallback) but with conversation history"""
        try:
            # Get conversation history
            messages = self.conversation_history.get_messages(channel_id)
            
            # Add current query if not already in messages
            if not messages or messages[-1]['content'] != query:
                messages.append({'role': 'user', 'content': query})
            
            response = self.anthropic.messages.create(
                max_tokens=2024,
                model='claude-3-7-sonnet-20250219',
                messages=messages
            )
            
            if response.content and response.content[0].type == 'text':
                response_text = response.content[0].text
                # Add assistant response to history
                self.conversation_history.add_message(channel_id, 'assistant', response_text)
                
                self.client.chat_postMessage(
                    channel=channel_id, 
                    text=response_text
                )
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            self.client.chat_postMessage(channel=channel_id, text=error_msg)

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        print(f"🔗 Connecting to {server_name}...")
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)

            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"✅ Connected to {server_name} with tools:", [t.name for t in tools])

            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        except Exception as e:
            print(f"❌ Failed to connect to {server_name}: {e}")
            raise  # Re-raise to be caught by the caller

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            # Try different possible locations for the config file
            config_paths = ["server_config.json", "mcp_project/server_config.json", "./server_config.json"]
            config_data = None
            
            for config_path in config_paths:
                try:
                    print(f"🔍 Looking for config at: {config_path}")
                    with open(config_path, "r") as file:
                        config_data = json.load(file)
                        print(f"✅ Found config at: {config_path}")
                        break
                except FileNotFoundError:
                    print(f"❌ Config not found at: {config_path}")
                    continue
            
            if not config_data:
                raise FileNotFoundError("server_config.json not found in any expected location")

            servers = config_data.get("mcpServers", {})
            print(f"🔧 Found {len(servers)} servers to connect to: {list(servers.keys())}")

            for server_name, server_config in servers.items():
                if server_name == 'slack':
                    server_config["env.SLACK_BOT_TOKEN"] = os.environ['SLACK_BOT_TOKEN']
                    server_config["env.SLACK_TEAM_ID"] = os.environ['SLACK_TEAM_ID']
                    server_config["env.SLACK_CHANNEL_IDS"] = os.environ['SLACK_CHANNEL_IDS']
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"❌ Error loading server configuration: {e}")
            raise

    async def process_slack_query(self, channel_id: str, query: str):
        """Process a query and send responses to Slack with conversation history"""
        try:
            # Get conversation history
            messages = self.conversation_history.get_messages(channel_id)
            
            # Add current query if not already in messages
            if not messages or messages[-1]['content'] != query:
                messages.append({'role': 'user', 'content': query})
            
            response = self.anthropic.messages.create(
                max_tokens=2024,
                model='claude-3-5-sonnet-20241022',
                tools=self.available_tools if self.available_tools else None,
                messages=messages
            )
            
            process_query = True
            full_assistant_response = ""
            
            while process_query:
                assistant_content = []
                for content in response.content:
                    if content.type == 'text':
                        print(f"Assistant: {content.text}")
                        full_assistant_response += content.text
                        # Send text response immediately to Slack
                        self.client.chat_postMessage(
                            channel=channel_id, 
                            text=content.text
                        )
                        assistant_content.append(content)
                        if len(response.content) == 1:
                            process_query = False
                            
                    elif content.type == 'tool_use':
                        assistant_content.append(content)
                        messages.append({'role': 'assistant', 'content': assistant_content})
                        
                        tool_id = content.id
                        tool_args = content.input
                        tool_name = content.name

                        print(f"Calling tool {tool_name} with args {tool_args}")
                        
                        # Send tool use notification to Slack
                        self.client.chat_postMessage(
                            channel=channel_id, 
                            text=f"🔧 Using tool: `{tool_name}` with arguments: `{json.dumps(tool_args, indent=2)}`"
                        )

                        # Call the tool
                        session = self.tool_to_session[tool_name]
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        
                        messages.append({
                            "role": "user", 
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result.content
                                }
                            ]
                        })
                        
                        response = self.anthropic.messages.create(
                            max_tokens=2024,
                            model='claude-3-5-sonnet-20241022',
                            tools=self.available_tools,
                            messages=messages
                        )

                        if (len(response.content) == 1 and 
                            response.content[0].type == "text"):
                            print(f"Assistant: {response.content[0].text}")
                            full_assistant_response += response.content[0].text
                            # Send final response to Slack
                            self.client.chat_postMessage(
                                channel=channel_id, 
                                text=response.content[0].text
                            )
                            process_query = False
            
            # Add the complete assistant response to conversation history
            if full_assistant_response:
                self.conversation_history.add_message(channel_id, 'assistant', full_assistant_response)

        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            print(error_msg)
            self.client.chat_postMessage(channel=channel_id, text=error_msg)

    async def cleanup(self):
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()

    def run(self):
        """Run the Flask app"""
        print("🚀 Starting MCP Slack Bot with Conversation Memory...")
        print("Mention the bot with 'start' to activate it in a channel")
        print("Say 'quit' or 'stop' to deactivate it in a channel")
        print("Say 'clear history' or 'reset' to clear conversation memory")
        self.app.run(debug=True, use_reloader=False)  # Disable reloader to avoid thread issues

# Create global bot instance
bot = MCP_SlackBot()

if __name__ == "__main__":
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down bot...")
        if bot.loop and bot.loop.is_running():
            bot.loop.call_soon_threadsafe(bot.loop.stop)
    finally:
        # Cleanup MCP resources
        if bot.loop:
            try:
                bot.loop.run_until_complete(bot.cleanup())
            except Exception as e:
                print(f"Error during cleanup: {e}")