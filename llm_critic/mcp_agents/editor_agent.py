from mcp_agent.agents.agent import Agent

editor = Agent(
    name="editor",
    instruction="""
You are the Editor. Your job is to assign the abstract to the reviewers for 4 dimensions: clarity, novelty, impact, and writing.
You will not evaluate directly. Instead, dispatch to specialized reviewers.
"""
)
