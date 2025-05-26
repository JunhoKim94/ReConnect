from typing import List, Dict, Optional, Any

from prompt import prompts

DATASET_TAGS = prompts.DATASET_TAGS
MCQ_EXAMPLE_TEMPLATE = prompts.MCQ_EXAMPLE_TEMPLATE
MCQ_WITH_KNOWLEDGE_EXAMPLE_TEMPLATE = prompts.MCQ_WITH_KNOWLEDGE_EXAMPLE_TEMPLATE
KNOWLEDGE_GENERATION_EXAMPLE_TEMPLATE = prompts.KNOWLEDGE_GENERATION_EXAMPLE_TEMPLATE
SHOT_TEMPLATES = prompts.SHOT_TEMPLATES


def prepare_sample_for_mcq(
    sample: Dict[str, Any],
    dataset_tag: str,
    examples: Optional[List[Dict[str, Any]]] = [],
    template_name: Optional[str] = "mcq",
    use_example_knowledge: Optional[bool] = False,
    questions_domain: Optional[str]=None,
    sample_knowledge: Optional[List[str]] = [],
) -> List[str]:
    """
    Prepare the prompt for the model.

    Parameters:
    - sample (Dict[str, Any]): Sample with the question and the options.
    - dataset_tag (str): Tag of the dataset.
    - examples (Optional[List[Dict[str, Any]]]): Examples to be used for fewshot learning.
    - template_name (Optional[str]): Template for the prompt; one of ["mcq", "mcq_with_kg"]. Default is "mcq".
    - use_example_knowledge (Optional[bool]): Whether to add the explanations for the options of the examples. Default is False.
    - sample_knowledge (Optional[List[str]]): Explanations for the options of the sample. Default is an empty list.

    Returns:
    - List[str]: Prompt for the model. The prompt consists of the conversation turns (user and assistant turns).

    Raises:
    - ValueError: If the template_name is "mcq" and use_example_knowledge is True.
    """
    if template_name == "mcq" and use_example_knowledge:
        raise ValueError("Template 'mcq' does not support the use of example knowledge.")

    # Get the template.
    template = SHOT_TEMPLATES[template_name]

    # Prepare the shots.
    shots = []

    for example in examples:
        question = example.get("question")
        answer = example.get("answer")

        # Preprocess the choices to display them in the prompt.
        # Each choice is displayed as "* {label}: {text}".
        choices = example.get("choices")
        choices = [f"* {c['label']}: {c['text']}".strip() for c in choices]
        choices = "\n".join(choices)

        # Preprocess the knowledge to display it in the prompt.
        # Each explanation is displayed as "* {explanation}".
        knowledge = example.get("knowledge")
        if use_example_knowledge and knowledge is not None:
            knowledge = [k if k.endswith(".") else f"{k}." for k in knowledge]
            knowledge = [f"* {k.strip()}" for k in knowledge]
            knowledge = "\n".join(knowledge)
        else:
            knowledge = None

        # Prepare the shot.
        shot = template.format(
            question=question,
            knowledge=knowledge,
            choices=choices,
            answer=answer,
        )

        # Add the user turn.
        shots.append(shot)

        # Add the assistant turn.
        shots.append(f"Answer: {answer}")

    # Prepare the sample.
    question = sample["question"]["stem"]
    choices = sample["question"]["choices"]
    choices = [f"* {c['label']}: {c['text']}".strip() for c in choices]
    choices = "\n".join(choices)

    # Preprocess the sample knowledge to display it in the prompt.
    # Each explanation is displayed as "* {explanation}".
    if sample_knowledge is not None:
        sample_knowledge = [k if k.endswith(".") else f"{k}." for k in sample_knowledge]
        sample_knowledge = [f"* {k.strip()}" for k in sample_knowledge]
        sample_knowledge = "\n".join(sample_knowledge)

    # Prepare the final shot for the user turn.
    final_shot = template.format(
        question=question,
        knowledge=sample_knowledge,
        choices=choices,
    )

    # Prepare the assistant turn.
    assistant_reply = "Answer: "

    if questions_domain and "custom" in dataset_tag:
        custom_num_choices = len(sample["question"]["choices"])
        dataset_tag_template = DATASET_TAGS[dataset_tag][template_name].format(
            questions_domain=questions_domain, 
            custom_num_choices=custom_num_choices
        )
    else:
        dataset_tag_template = DATASET_TAGS[dataset_tag][template_name]
    # Build the conversation turns.
    prompt = (
        [
            dataset_tag_template,
            "Yes, I understand. Please provide the question and the possible options.",
        ]
        + shots
        + [final_shot, assistant_reply]
    )

    # Return the prompt / conversation turns.
    return prompt


def prepare_sample_for_knowledge_generation(
    sample: Dict[str, Any],
    dataset_tag: str,
    shot_samples: Optional[List[Dict[str, Any]]]=[],
    template_name: Optional[str]="knowledge_generation",
) -> List[str]:
    """
    Prepare the prompt for the model.

    Parameters:
    - sample (Dict[str, Any]): Sample with the question and the options.
    - dataset_tag (str): Tag of the dataset.
    - shot_samples (Optional[List[Dict[str, Any]]]): Samples to be used for fewshot learning.
    - template_name (Optional[str]): Template for the prompt; one of ["knowledge_generation"]. Default is "knowledge_generation".
    
    Returns:
    - List[str]: Prompt for the model. The prompt consists of the conversation turns (user and assistant turns).
    """
    template = SHOT_TEMPLATES[template_name]
    shots = []

    for shot in shot_samples:
        question = shot.get("question")
        choices = shot.get("choices")
        knowledge = shot.get("knowledge")

        # Preprocess the choices.
        choices = [f"* {choice['text']}" for choice in choices]
        choices = "\n".join(choices)

        # Preprocess the knowledge.
        if knowledge is not None:
            knowledge = [k if k.endswith(".") else f"{k}." for k in knowledge]
            knowledge = [f"* {k}" for k in knowledge]
            knowledge = "\n".join(knowledge)
            knowledge = f"Explanations:\n{knowledge}"
        else:
            knowledge = ""

        shot = template.format(question=question, choices=choices)

        shots.append(shot)  # User turn.
        shots.append(knowledge)  # Assistant turn.

    question = sample["question"]["stem"]
    choices = sample["question"]["choices"]
    choices = [f"* {choice['text']}" for choice in choices]
    choices = "\n".join(choices)

    final_shot = template.format(question=question, choices=choices)

    # Force the assistant to provide the explanations.
    assistant_reply = "Explanations:\n* "
    

    # Build the conversation turns.
    # Check if the dataset tag contains "custom" and adjust the prompt accordingly.
    if "custom" in dataset_tag:
        custom_num_choices = len(sample["question"]["choices"])
        dataset_tag_template = DATASET_TAGS[dataset_tag][template_name].format(custom_num_choices=custom_num_choices)
    else:
        dataset_tag_template = DATASET_TAGS[dataset_tag][template_name]

    # Build the conversation turns.
    prompt = (
        [
            dataset_tag_template,
            "Yes, I understand. Please provide the question and the possible options.",
        ]
        + shots
        + [final_shot, assistant_reply]
    )

    return prompt
