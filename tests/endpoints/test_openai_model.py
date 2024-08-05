import pytest

from lighteval.models.endpoints import OpenAIModel
from lighteval.tasks.requests import GreedyUntilRequest, LoglikelihoodRequest
from lighteval.evaluator import EvaluationTracker, evaluate
from lighteval.models.model_config import EndpointConfig, EnvConfig
from lighteval.models.model_loader import load_model
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.tasks_prompt_formatting import arc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import RequestType, Doc, TaskExampleId


def test_openai_model_api(openai_model: OpenAIModel):
    requests = [
        GreedyUntilRequest("test_task", 0, 0, "How many hands does human have?", [], 5, num_samples=1),
        GreedyUntilRequest("test_task", 0, 0, "How many eyes does human have?", [], 5, num_samples=1)
    ]
    returns = openai_model.greedy_until(requests)
    assert len(returns) == 2
    assert all((type(r.result) is str) and len(r.result) for r in returns)

    gpt35t =  OpenAIModel("gpt-3.5-turbo-instruct")

    requests.append(
        GreedyUntilRequest("test_task", 0, 0, "How many ears does human have?", [], 5, num_samples=1, use_logits=True),
    )

    with pytest.raises(ValueError, match=r"OpenAI models could not process requests with `use_logits=True`"):
        gpt35t.greedy_until(requests)

    requests = [
        LoglikelihoodRequest("test_task", 0, 0, "How many hands does human have?", "Two"),
        LoglikelihoodRequest("test_task", 0, 0, "How many eyes does human have?", "Two")
    ]
    returns = openai_model.loglikelihood(requests)
    assert len(returns) == 2
    
    with pytest.raises(ValueError, match=r"OpenAI models could not be evaluated by non-generative metrics"):
        gpt35t.loglikelihood(requests)
    
    openai_model.tok_encode("Hi there")


def test_openai_model_integration():
    evaluation_tracker = EvaluationTracker()
    model_config = EndpointConfig("openai", "davinci-002")
    model, _ = load_model(config=model_config, env_config=EnvConfig())

    task_config = LightevalTaskConfig("test", arc, "", "", [Metrics.loglikelihood_acc])
    task = LightevalTask("test", task_config)
    task_dict = {"custom|test": task}
    evaluation_tracker.task_config_logger.log(task_dict)
    doc = Doc("Who is the GOAT?", ["CR7", "Messi", "Pele", "Zizou"], gold_index=3)
    doc.ctx = "Who is the GOAT?"
    docs = {TaskExampleId("custom|test|0", "0_0"): doc}
    requests_dict = task.construct_requests(doc, doc.ctx, "0_0", "custom|test|0")
    # Because `task.construct_requests` has empty entries causing error in `evaluate`` currently.
    requests_dict = {RequestType.LOGLIKELIHOOD: requests_dict[RequestType.LOGLIKELIHOOD]}

    evaluation_tracker = evaluate(
        lm=model,
        requests_dict=requests_dict,
        docs=docs,
        task_dict=task_dict,
        override_bs=1,
        evaluation_tracker=evaluation_tracker,
    )
    evaluation_tracker.metrics_logger.aggregate(task_dict=task_dict)
    evaluation_tracker.details_logger.aggregate()
    model.cleanup()
    evaluation_tracker.generate_final_dict()