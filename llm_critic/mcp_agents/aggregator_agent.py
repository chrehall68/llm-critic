from mcp_agent.agents.agent import Agent

aggregator = Agent(
    name="aggregator",
    instruction="""
Aggregate the 4 reviewer scores.
If avg >= 3.5, say "Conference Ready", else "Needs Improvement".
Respond JSON: {"average_score": float, "decision": str}
"""
)
