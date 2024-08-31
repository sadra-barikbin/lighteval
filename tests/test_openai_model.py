import pytest
from typing import TypeAlias
from collections import defaultdict

from lighteval.models.endpoints import OpenAIModel
from lighteval.tasks.requests import GreedyUntilRequest, LoglikelihoodRequest
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_config import EndpointConfig, EnvConfig
from lighteval.models.model_loader import load_model
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig
from lighteval.tasks.default_prompts import arc
from lighteval.metrics.metrics import Metrics
from lighteval.tasks.requests import RequestType, Doc, Request


@pytest.fixture(scope='module', params=["davinci-002", "gpt-3.5-turbo-0613"])
def openai_model(request):
    model =  OpenAIModel(request.param)
    yield model
    model.cleanup()


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

RequestDict: TypeAlias = dict[RequestType, list[Request]]


class TestOpenAIModel:
    @pytest.fixture
    def task(self) -> LightevalTask:
        eval_docs = [
            Doc(
                query="Tell me:\n\nHow are you?",
                choices=["Fine, thanks!", "Not bad!"],
                instruction="Tell me:\n\n",
                gold_index=0,
            ),
            Doc(
                query="Tell me:\n\nComment vas-tu?",
                choices=["Ca va! Merci!", "Comme ci, comme ça"],
                instruction="Tell me:\n\n",
                gold_index=0,
            ),
        ]
        fewshot_docs = [
            Doc(
                query="Tell me:\n\nكيف حالك؟",
                choices=["جيد شكراً!", "ليس سيئًا!"],
                instruction="Tell me:\n\n",
                gold_index=0,
            ),
            Doc(
                query="Tell me:\n\nWie geht es dir?",
                choices=["Gut, danke!", "Nicht schlecht!"],
                instruction="Tell me:\n\n",
                gold_index=0,
            ),
        ]
        task_config = LightevalTaskConfig(
            name="test",
            prompt_function=lambda _: _,
            hf_repo="",
            hf_subset="",
            metric=[Metrics.loglikelihood_acc, Metrics.exact_match, Metrics.byte_perplexity],
            generation_size=5,
            stop_sequence=[],
        )
        task = LightevalTask("test", task_config)
        task._docs = eval_docs
        task._fewshot_docs = fewshot_docs
        return task

    @pytest.fixture
    def zero_shot_request_dict(self, task: LightevalTask) -> RequestDict:
        result = defaultdict(list)
        for i, doc in enumerate(task.eval_docs()):
            if i % 2 == 0:
                context = [ChatCompletionInputMessage(role="user", content=doc.query)]
            else:
                context = doc.query
            doc_result = task.construct_requests(doc, context, f"{i}_0", "custom|test|0")
            for req_type in doc_result:
                result[req_type].extend(doc_result[req_type])
        return result

    def test_greedy_until(self, zero_shot_request_dict: RequestDict, tgi_model: TGIModel):
        returns = tgi_model.greedy_until(zero_shot_request_dict[RequestType.GREEDY_UNTIL])
        assert len(returns) == 2
        assert all(r.result is not None for r in returns)

    def test_loglikelihood(self, zero_shot_request_dict: RequestDict, tgi_model: TGIModel):
        returns = tgi_model.loglikelihood(zero_shot_request_dict[RequestType.LOGLIKELIHOOD])
        assert len(returns) == 4
        assert all(r.result is not None for r in returns)

        returns = tgi_model.loglikelihood_rolling(zero_shot_request_dict[RequestType.LOGLIKELIHOOD_ROLLING])
        assert len(returns) == 2
        assert all(r.result is not None for r in returns)

    @pytest.mark.parametrize("num_fewshot", [0, 2])
    @pytest.mark.parametrize("use_chat_template", [False, True])
    def test_integration(self, task: LightevalTask, tgi_model: TGIModel, num_fewshot: int, use_chat_template: bool):
        env_config = EnvConfig(token=TOKEN, cache_dir=CACHE_PATH)
        evaluation_tracker = EvaluationTracker()
        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            use_chat_template=use_chat_template,
        )

        with patch("lighteval.pipeline.Pipeline._init_tasks_and_requests"):
            pipeline = Pipeline(
                tasks=f"custom|test|{num_fewshot}|0",
                pipeline_parameters=pipeline_params,
                evaluation_tracker=evaluation_tracker,
                model=tgi_model,
            )
        task_dict = {"custom|test": task}
        evaluation_tracker.task_config_logger.log(task_dict)
        fewshot_dict = {"custom|test": [(num_fewshot, False)]}
        pipeline.task_names_list = ["custom|test"]
        pipeline.task_dict = task_dict
        pipeline.fewshot_dict = fewshot_dict
        requests, docs = create_requests_from_tasks(
            task_dict=task_dict,
            fewshot_dict=fewshot_dict,
            num_fewshot_seeds=pipeline_params.num_fewshot_seeds,
            lm=tgi_model,
            max_samples=pipeline_params.max_samples,
            evaluation_tracker=evaluation_tracker,
            use_chat_template=use_chat_template,
            system_prompt=pipeline_params.system_prompt,
        )
        pipeline.requests = requests
        pipeline.docs = docs
        evaluation_tracker.task_config_logger.log(task_dict)

        pipeline.evaluate()
