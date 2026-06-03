from ccindex.agents.base import AgentAdapter
from ccindex.agents.claude_code import ClaudeCodeAdapter
from ccindex.agents.gemini_cli import GeminiCliAdapter
from ccindex.agents.antigravity import AntigravityAdapter

ADAPTERS: dict[str, type[AgentAdapter]] = {
    "claude-code": ClaudeCodeAdapter,
    "gemini-cli": GeminiCliAdapter,
    "antigravity": AntigravityAdapter,
}
