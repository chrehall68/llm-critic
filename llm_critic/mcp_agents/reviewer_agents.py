from mcp_agent.agents.agent import Agent

reviewer_clarity = Agent(
    name="reviewer_clarity",
    instruction="""
Evaluate clarity. Score 1–5. Explain. 
Respond JSON: {"score": <int>, "comments": ["...", "..."]}
"""
)

reviewer_novelty = Agent(
    name="reviewer_novelty",
    instruction="""
Evaluate novelty. Score 1–5. Explain. 
Respond JSON: {"score": <int>, "comments": ["...", "..."]}
"""
)

reviewer_impact = Agent(
    name="reviewer_impact",
    instruction="""
Evaluate impact. Score 1–5. Explain. 
Respond JSON: {"score": <int>, "comments": ["...", "..."]}
"""
)

reviewer_writing = Agent(
    name="reviewer_writing",
    instruction="""
Evaluate writing quality. Score 1–5. Explain. 
Respond JSON: {"score": <int>, "comments": ["...", "..."]}
"""
)

reviewers = [reviewer_clarity, reviewer_novelty, reviewer_impact, reviewer_writing]
