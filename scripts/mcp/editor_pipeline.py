import asyncio
from mcp_agent.app import MCPApp
from llm_critic.mcp_agents.editor_agent import editor
from llm_critic.mcp_agents.reviewer_agents import reviewers
from llm_critic.mcp_agents.aggregator_agent import aggregator

app = MCPApp(name="llm_critic_editor")

async def main(abstract_text: str):
    async with app.run():
        await app.invoke(editor.name, {"abstract": abstract_text})

        responses = await asyncio.gather(*[
            app.invoke(agent.name, {"abstract": abstract_text}) for agent in reviewers
        ])

        merged = {"reviews": responses}
        final = await app.invoke(aggregator.name, merged)

        print("\n=== Final Decision ===")
        print(final)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python scripts/mcp/editor_pipeline.py examples/test_abstract.txt")
        sys.exit(1)

    text = open(sys.argv[1], "r").read()
    asyncio.run(main(text))
