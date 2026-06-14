from __future__ import annotations

from pi.agent import AgentTool
from pi.coding import create_all_tools

def create_all_coding_tools(cwd: str) -> list[AgentTool]:
    return list(create_all_tools(cwd).values())
