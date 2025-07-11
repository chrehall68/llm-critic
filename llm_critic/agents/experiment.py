# simulate the peer review process
# except just for the abstract
# it'll work by having N reviewers review each abstract
# and then using a final "editor" to decide on the final decision based
# on the reviewers' decisions
from dataclasses import dataclass
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion
import asyncio
import os
from tqdm import tqdm
from datasets import Dataset

from llm_critic.core.llm_critic import was_correct
from .prompts import make_initial_editor_prompt, AGENTS, make_final_editor_prompt


@dataclass
class PeerReviewResult:
    initial_editor_response: str
    reviewers: dict[str, str]
    final_decision: str


client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT", None),  # optional
    base_url=os.getenv(
        "OPENAI_BASE_URL", None
    ),  # optional; provide if you're not using OpenAI
)


async def try_request(args: dict):
    good = False
    i = 0
    result: ChatCompletion = None
    while not good:
        try:
            result = await client.chat.completions.create(**args)
            good = True
        except RateLimitError:
            await asyncio.sleep(2**i)
            i += 1
    return result


async def peer_review(
    abstract: str, reviewer_models: dict[str, str], editor_model: str
) -> PeerReviewResult:
    """
    Runs the agentic peer review process
    """
    reviewer_name_to_reviewer = {agent.name: agent for agent in AGENTS}
    # first, ask the editor, with the initial prompt,
    # to assign reviewers
    editor_response = await try_request(
        dict(
            model=editor_model,
            messages=[
                {
                    "role": "user",
                    "content": make_initial_editor_prompt() + abstract,
                },
            ],
            temperature=1,
            max_tokens=256,
        )
    )
    # parse the response
    reviewers = editor_response.choices[0].message.content.split(",")
    reviewers = [r.strip() for r in reviewers]

    # forward to reviewers
    tasks = []
    for reviewer in reviewers:
        if reviewer in reviewer_name_to_reviewer:  # just in case
            tasks.append(
                try_request(
                    dict(
                        model=reviewer_models[reviewer],
                        messages=[
                            {
                                "role": "user",
                                "content": reviewer_name_to_reviewer[reviewer].prompt
                                + abstract,
                            },
                        ],
                        temperature=1,
                        max_tokens=2048,
                    )
                )
            )
    if not tasks:
        # no reviewers were assigned, so just wrong
        return PeerReviewResult(
            initial_editor_response=editor_response.choices[0].message.content,
            reviewers={},
            final_decision="",
        )
    reviewer_recommendations = await asyncio.gather(*tasks)

    # finally ask the final "editor"
    editor_recommendation = await try_request(
        dict(
            model=editor_model,
            messages=[
                {
                    "role": "user",
                    "content": make_final_editor_prompt(
                        {
                            name: r.choices[0].message.content
                            for name, r in zip(reviewers, reviewer_recommendations)
                        }
                    )
                    + abstract,
                }
            ],
            temperature=0.7,
            max_tokens=256,
        )
    )
    return PeerReviewResult(
        initial_editor_response=editor_response.choices[0].message.content,
        reviewers={
            name: r.choices[0].message.content
            for name, r in zip(reviewers, reviewer_recommendations)
        },
        final_decision=editor_recommendation.choices[0].message.content,
    )


async def run_experiment_async(
    ds: Dataset,
    start: int,
    end: int,
    reviewer_models: dict[str, str],
    editor_model: str,
    num_concurrent: int = 10,
):
    sema = asyncio.Semaphore(num_concurrent)

    async def throttler(f):
        async with sema:
            return await f

    async def review_and_score(i: int):
        abstract = ds[i]["abstract"]
        r = await peer_review(abstract, reviewer_models, editor_model)
        correct = was_correct(r.final_decision, ds[i])
        return {
            "abstract": abstract,
            "reviewer_recommendations": r.reviewers,
            "editor_recommendation": r.final_decision,
            "initial_editor_response": r.initial_editor_response,
            "was_correct": correct,
        }

    requests = [throttler(review_and_score(i)) for i in range(start, end)]
    results: list[dict] = []
    correct = 0
    for result in (prog := tqdm(asyncio.as_completed(requests), total=len(requests))):
        results.append(await result)
        correct += results[-1]["was_correct"]
        prog.set_postfix({"correct": correct, "runningAvg": correct / len(results)})

    return results


def run_experiment(
    ds: Dataset,
    start: int,
    end: int,
    reviewer_models: dict[str, str],
    editor_model: str,
    num_concurrent: int = 10,
):
    """
    Runs the agentic peer review experiment

    Arguments:
        - ds: Dataset - the dataset
        - start: int - the index of the first sample to evaluate on, inclusive
        - end: int - the index of the last sample to evaluate on, exclusive
        - reviewer_models: dict[str, str] - the models to use for reviewers
        - editor_model: str - the model to use for the editor
        - num_concurrent: int = 10 - the number of concurrent agentic peer review
          processes to run at once
    """
    return asyncio.run(
        run_experiment_async(
            ds, start, end, reviewer_models, editor_model, num_concurrent
        )
    )
