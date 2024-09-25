import re
import random
import pytest

from lighteval.models.model_output import RegexAnswerExtractor
from lighteval.evaluator import EvaluationTracker, evaluate
from lighteval.metrics import Metrics
from lighteval.models.base_model import BaseModel
from lighteval.models.model_config import BaseModelConfig, EnvConfig
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig, create_requests_from_tasks
from lighteval.tasks.requests import (
    Doc,
)


def test_answer_extractor():
    # MMLU-Pro
    extractor = RegexAnswerExtractor(
        [re.compile(r"answer is \(?\(([A-J])\)?\)"),
         re.compile(r"\.*\[aA\]nswer:\s*\(([A-J])\)")],
         fallback="random")
    
    assert extractor("answer is (C)", ["A", "B", "C", "D"]) == "C"

    random.seed(41)
    fallback_choice = random.choice(["A", "B", "C", "D"])
    random.seed(41)
    assert extractor("answer is (F)", ["A", "B", "C", "D"]) == fallback_choice

    extractor.fallback = "keep"
    assert extractor("I don't know", ["A", "B", "C", "D"]) == "I don't know"

    extractor.fallback = 0
    assert extractor("I don't know", ["A", "B", "C", "D"]) == "A"


@pytest.fixture(scope="module")
def base_model() -> BaseModel:
    config = BaseModelConfig("hf-internal-testing/tiny-random-LlamaForCausalLM")
    return BaseModel(config, EnvConfig())


@pytest.fixture
def task() -> LightevalTask:
    eval_docs = [
        Doc(
            query="Tell me:\n\nHow many eyes do you have?",
            choices=["2", "3"],
            instruction="Tell me:\n\n",
            gold_index=0,
        ),
        Doc(
            query="Tell me:\n\nHow many hands do we have?",
            choices=["2", "3"],
            instruction="Tell me:\n\n",
            gold_index=0,
        ),
    ]
    task_config = LightevalTaskConfig(
        name="test",
        prompt_function="arc",
        hf_repo="",
        hf_subset="",
        metric=["exact_match"],
        answer_extractor=RegexAnswerExtractor([r"\w", r"\d"]),
        generation_size=1,
        stop_sequence=[],
    )
    task = LightevalTask("test", task_config)
    task._docs = eval_docs
    return task


def test_integration(task: LightevalTask, base_model: BaseModel):
    evaluation_tracker = EvaluationTracker()
    task_dict = {"custom|test": task}
    evaluation_tracker.task_config_logger.log(task_dict)
    requests_dict, docs = create_requests_from_tasks(
        task_dict=task_dict,
        fewshot_dict={"custom|test": [(0, False)]},
        num_fewshot_seeds=0,
        lm=base_model,
        max_samples=1,
        evaluation_tracker=evaluation_tracker,
        use_chat_template=False,
        system_prompt=None,
    )

    evaluation_tracker = evaluate(
        lm=base_model,
        requests_dict=requests_dict,
        docs=docs,
        task_dict=task_dict,
        override_bs=1,
        evaluation_tracker=evaluation_tracker,
    )
    evaluation_tracker.metrics_logger.aggregate(task_dict=task_dict)
    evaluation_tracker.details_logger.aggregate()
    evaluation_tracker.generate_final_dict()