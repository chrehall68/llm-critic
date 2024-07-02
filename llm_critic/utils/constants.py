# constants
LABEL_MAP = {0: "Reject", 1: "Accept"}
GENERATION_ARGS = {
    "max_new_tokens": 4,
    "temperature": 0.7,
    "do_sample": True,
    "top_k": 10,
}
CHAT_TEMPLATES = {
    "llama": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\nReviewer decision:' }}{% endif %}",
    "gemma": """{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\nReviewer decision:'}}{% endif %}""",
    "galactica": """{{ 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n' }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '### Instruction:\n' + message['content'].strip() + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Response:'  + message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### Response: Reviewer decision:' }}{% endif %}""",
}
MODEL_MAP = {
    "gemma": "google/gemma-7b-it",
    "galactica": "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
}
SYSTEM_SUPPORTED = {"llama": True, "galactica": False, "gemma": False, "openai": True}
MAX_LEN = 2000
LAYER_MAP = {
    "llama": (lambda model: model.model, "embed_tokens"),
    "gemma": (lambda model: model.model, "embed_tokens"),
    "galactica": (lambda model: model.model.decoder, "embed_tokens"),
}
REJECT = 0
ACCEPT = 1
# REJECT = 0, ACCEPT = 1
# LLAMA 3
# " Reject" = 88393
# " Accept" = 21496
# Gemma
# " Accept" = 38601
# " Reject" = 140754
# Galactica
# " Accept" = 31837
# " Re" = 1372
TOKEN_MAP = {
    "llama": {0: 88393, 1: 21496},
    "gemma": {0: 140754, 1: 38601},
    "galactica": {0: 1372, 1: 31837},
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a NeurIPS reviewer with many years of experience reviewing papers. "
    + "You can tell whether a paper will be accepted just by looking at its abstract.\n"
    + 'For example, given "Abstract: This paper is an example rejected abstract", you might respond "Reviewer decision: Reject"\n'
    + 'As another example, given "Abstract: This paper is an example accepted abstract", you might respond "Reviewer decision: Accept"\n'
)
