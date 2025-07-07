from dataclasses import dataclass


@dataclass
class Agent:
    name: str
    description: str
    # add the abstract to the prompt
    # for proper formatting
    prompt: str


def make_agent(name: str, field: str, knowledge_areas: list[str]) -> Agent:
    def make_reviewer_prompt(field: str, knowledge_areas: list[str]) -> str:
        joined = ", ".join(knowledge_areas + ["and more"])
        return (
            f"""You are an experienced and well-respected academic in the field of {field}. You have been working in the field for many years. """
            + f"""As such, you are highly knowledgeable about {joined}.\n"""
            + """You have been asked to review a paper. Please evaluate the following abstract and give your thoughts on whether the paper is likely to be accepted based on the technical content previewed in the abstract."""
            + """Please limit your response to three paragraphs.\n"""
            + """Abstract: """
        )

    description = f"A reviewer experienced in the field of {field}"
    return Agent(name, description, make_reviewer_prompt(field, knowledge_areas))


AGENTS = [
    make_agent(
        "ReinforcementLearning_Reviewer",
        "reinforcement learning",
        [
            "On-Policy and Off-Policy algorithms",
            "deep learning architectures",
            "imitation learning",
        ],
    ),
    make_agent(
        "DeepLearning_Reviewer",
        "deep learning",
        [
            "neural network architectures",
            "CNNs",
            "RNNs",
            "transformers",
            "unsupervised learning",
            "supervised learning",
            "self-supervised learning",
            "autoencoders",
        ],
    ),
    make_agent(
        "NaturalLanguageProcessing_Reviewer",
        "natural language processing",
        [
            "tokenization",
            "word embeddings",
            "sentence embeddings",
            "transformers",
            "information retrieval",
        ],
    ),
    make_agent(
        "ComputerVision_Reviewer",
        "computer vision",
        [
            "image classification",
            "image segmentation",
            "object detection",
            "image captioning",
            "image generation",
            "diffusion models",
            "generative adversarial networks",
        ],
    ),
    make_agent(
        "AudioProcessing_Reviewer",
        "audio processing",
        [
            "speech recognition",
            "speech synthesis",
            "speech-to-text",
            "text-to-speech",
            "speech enhancement",
            "speech denoising",
            "voice conversion",
            "voice cloning",
            "music generation",
        ],
    ),
    make_agent(
        "Robotics_Reviewer",
        "robotics",
        [
            "robotics control",
            "robotics perception",
            "robotics planning",
            "robotics navigation",
            "robotics localization",
            "robotics manipulation",
            "robotics learning",
            "robotics safety",
        ],
    ),
    make_agent(
        "Alignment_Reviewer",
        "AI safety, alignment, and ethics",
        [
            "fairness and bias",
            "explainability",
            "interpretability",
            "alignment",
            "robustness",
        ],
    ),
    make_agent(
        "Mathematics_Reviewer",
        "mathematics, statistics, and optimization",
        [
            "mathematics",
            "statistics",
            "optimization",
            "linear algebra",
            "probability theory",
            "differential equations",
            "integral equations",
        ],
    ),
]


def make_initial_editor_prompt() -> str:
    reviewers_formatted = "\n".join(f"{r.name} | {r.description}" for r in AGENTS)
    return (
        """You are an experienced researcher with a PhD and many years in the field. """
        + """You are the initial editor of a paper. Given an abstract, your task is to forward it to the appropriate reviewers. """
        + """Choose at least one reviewer. """
        + """You will output only the reviewers' names, exactly as they appear, as a comma-separated list.\n"""
        + """Reviewer | Description\n"""
        + """-------- | --------\n"""
        + reviewers_formatted
        + "\n\nAbstract: "
    )


def make_final_editor_prompt(reviewer_responses: dict[str, str]) -> str:
    """
    Returns the prompt for the final editor
    reviewer_responses should map agent name to response
    """
    formatted_responses = "\n--------\n".join(
        f"{r}: {resp}" for r, resp in reviewer_responses.items()
    )
    return (
        """You are an experienced researcher with a PhD and many years in the field. """
        + """You are the final editor of a paper. You have forwarded the paper to several reviewers and received their comments. """
        + """You will now decide whether the paper should be accepted or rejected to NeurIPS. Only output "Accept" or "Reject".\n"""
        + """Reviewer responses:\n"""
        + """--------\n"""
        + formatted_responses
        + """--------\n"""
        + "\nAbstract: "
    )
