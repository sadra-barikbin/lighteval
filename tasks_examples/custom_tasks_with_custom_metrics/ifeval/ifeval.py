import instructions_registry
import numpy as np

from lighteval.metrics.utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetricGrouping,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


# We create the task config
ifeval = LightevalTaskConfig(
    name="ifeval",
    prompt_function="ifeval_prompt",
    hf_repo="wis-k/instruction-following-eval",
    hf_subset="train",
    metric=["ifeval_metric"],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,  # to check
    stop_sequence=["\n"],  # to check
)


def ifeval_prompt(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[],  # very specific task where there are no precise outputs but instead we test if the format obeys rules
        gold_index=-1,
        instruction="",
        specific={"instruction_id_list": line["instruction_id_list"], "kwargs": line["kwargs"]},
    )


submetric_names = [
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
]


def ifeval_metric(predictions: list[str], formatted_doc: Doc, **kwargs) -> float:
    response = predictions[0]

    # Strict instructions
    instruction_list = formatted_doc.specific["instructions_id_list"]
    kwargs = formatted_doc.specific["kwargs"]
    prompt = formatted_doc.query

    # Loose instructions
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    is_following_list_strict = []
    is_following_list_loose = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Remove None values from kwargs to avoid unexpected keyword argument errors in build_description method.
        kwargs = {k: v for k, v in kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        # Strict
        if response.strip() and instruction.check_following(response):
            is_following_list_strict.append(True)
        else:
            is_following_list_strict.append(False)

        # Loose
        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list_loose.append(is_following)

    return {
        "prompt_level_strict_acc": int(all(is_following_list_strict)),
        "inst_level_strict_acc": np.mean(is_following_list_strict),
        "prompt_level_loose_acc": int(all(is_following_list_loose)),
        "inst_level_loose_acc": np.mean(is_following_list_loose),
    }


ifeval_metrics = SampleLevelMetricGrouping(
    metric=submetric_names,
    higher_is_better={n: True for n in submetric_names},
    category=MetricCategory.GENERATIVE,
    use_case=MetricUseCase.ACCURACY,
    sample_level_fn=ifeval_metric,
    corpus_level_fn=np.mean,
)
