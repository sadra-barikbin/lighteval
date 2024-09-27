# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import defaultdict
from typing import TypeAlias

import pytest
from huggingface_hub import ChatCompletionInputMessage, TextGenerationInputGrammarType
from pydantic import BaseModel

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.endpoints import OpenAIModel
from lighteval.evaluator import EvaluationTracker, evaluate
from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig, create_requests_from_tasks
from lighteval.tasks.requests import (
    Doc,
    GreedyUntilRequest,
    Request,
    RequestType,
    LoglikelihoodRequest,
    LoglikelihoodRollingRequest,
)


@pytest.fixture(scope="module", params=['gpt-4o-mini'])
def openai_model(request):
    model = OpenAIModel(request.param)
    yield model
    model.cleanup()


RequestDict: TypeAlias = dict[RequestType, list[Request]]

class ResponseFormat(BaseModel):
    expression: str
    french_translation: str
    arabic_translation: str


class TestOpenAIModel:
    @pytest.fixture
    def structured_generation_task(self) -> LightevalTask:
        eval_docs = [
            Doc(
                query="Respond in JSON:\n\nWhat's the translation of 'I love you' in French and Arabic?",
                choices=[ResponseFormat(expression="I love you",
                                        french_translation="Je t'aime",
                                        arabic_translation="أحبك").model_dump_json()],
                instruction="Respond in JSON:\n\n",
                gold_index=0,
            ),
        ]
        fewshot_docs = [
            Doc(
                query="Respond in JSON:\n\nWhat's the translation of 'light' in French and Arabic?",
                choices=[ResponseFormat(expression="light",
                                        french_translation="lumière",
                                        arabic_translation="نور").model_dump_json()],
                instruction="Respond in JSON:\n\n",
                gold_index=0,
            ),
            Doc(
                query="Respond in JSON:\n\nWhat's the translation of 'friend' in French and Arabic?",
                choices=[ResponseFormat(expression="friend",
                                        french_translation="amy",
                                        arabic_translation="صدیق").model_dump_json()],
                instruction="Respond in JSON:\n\n",
                gold_index=0,
            ),
        ]
        task_config = LightevalTaskConfig(
            name="test_structured",
            prompt_function="arc",
            hf_repo="",
            hf_subset="",
            metric=["exact_match"],
            generation_grammar=TextGenerationInputGrammarType(type="json", value=ResponseFormat.model_json_schema()),
            stop_sequence=[],
        )
        task = LightevalTask("test_structured", task_config)
        task._docs = eval_docs
        task._fewshot_docs = fewshot_docs
        return task

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
            prompt_function="arc",
            hf_repo="",
            hf_subset="",
            metric=["exact_match"],
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
            context = [ChatCompletionInputMessage(role="user", content=doc.query)]
            doc_result = task.construct_requests(doc, context, f"{i}_0", "custom|test|0")
            for req_type in doc_result:
                result[req_type].extend(doc_result[req_type])
        return result

    def test_greedy_until(self, zero_shot_request_dict: RequestDict, openai_model: OpenAIModel):
        returns = openai_model.greedy_until(zero_shot_request_dict[RequestType.GREEDY_UNTIL])
        assert len(returns) == 2
        assert all(r.result is not None for r in returns)
        request = GreedyUntilRequest(
            "test_task", 0, 0,
            [ChatCompletionInputMessage(role="user",content="How many ears does human have?")],
            [], 5, None, num_samples=1, use_logits=True
        )
        response = openai_model.greedy_until([request])[0]
        assert response.generated_tokens is not None
        assert response.result is not None
        assert response.logits is not None

        request.generation_grammar = TextGenerationInputGrammarType(type="regex", value=r"\d+")
        with pytest.raises(ValueError, match="OpenAI models don't support structured generation with regex"):
            openai_model.greedy_until([request])

    def test_loglikelihood(self, openai_model: OpenAIModel):
        requests = [LoglikelihoodRequest("test_task", 0, 0, "Hi there!", "Hello there!")]
        with pytest.raises(ValueError, match=r"OpenAI models could not be evaluated by non-generative metrics"):
            openai_model.loglikelihood(requests)
        requests = [LoglikelihoodRollingRequest("test_task", 0, 0, "Hi there!", "Hello there!")]
        with pytest.raises(ValueError, match=r"OpenAI models could not be evaluated by non-generative metrics"):
            openai_model.loglikelihood_rolling(requests)

    @pytest.mark.parametrize("num_fewshot", [0, 2])
    def test_integration(
        self, task: LightevalTask, structured_generation_task: LightevalTask, openai_model: OpenAIModel, num_fewshot: int):
        evaluation_tracker = EvaluationTracker()
        task_dict = {"custom|test": task, "custom|test_structured": structured_generation_task}
        evaluation_tracker.task_config_logger.log(task_dict)
        requests_dict, docs = create_requests_from_tasks(
            task_dict=task_dict,
            fewshot_dict={"custom|test": [(num_fewshot, False)], "custom|test_structured": [(num_fewshot, False)]},
            num_fewshot_seeds=0,
            lm=openai_model,
            max_samples=None,
            evaluation_tracker=evaluation_tracker,
            use_chat_template=True,
            system_prompt=None,
        )

        evaluation_tracker = evaluate(
            lm=openai_model,
            requests_dict=requests_dict,
            docs=docs,
            task_dict=task_dict,
            override_bs=1,
            evaluation_tracker=evaluation_tracker,
        )
        evaluation_tracker.metrics_logger.aggregate(task_dict=task_dict)
        evaluation_tracker.details_logger.aggregate()
        result = evaluation_tracker.generate_final_dict()
        assert result["results"][f"custom:test_structured:{num_fewshot}"]["em"] == 1
