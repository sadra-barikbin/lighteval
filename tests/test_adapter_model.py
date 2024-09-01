import os
import pytest
import tempfile
import yaml
from argparse import Namespace

from lighteval.tasks.lighteval_task import LightevalTask, LightevalTaskConfig, create_requests_from_tasks
from lighteval.models.model_config import  EnvConfig, create_model_config
from lighteval.models.model_loader import load_model
from lighteval.tasks.requests import Doc
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.evaluator import evaluate


TOKEN = os.environ.get("HF_TOKEN")
CACHE_PATH = os.getenv("HF_HOME", ".")


class TestAdapterModel:
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
            metric=["loglikelihood_acc", "exact_match", "byte_perplexity"],
            generation_size=5,
            stop_sequence=[],
        )
        task = LightevalTask("test", task_config)
        task._docs = eval_docs
        task._fewshot_docs = fewshot_docs
        return task
    
    def test_integration(self, task: LightevalTask):
        evaluation_tracker = EvaluationTracker()
        task_dict = {"custom|test": task}
        evaluation_tracker.task_config_logger.log(task_dict)
        fewshot_dict = {"custom|test": [(0, False)]}
        with tempfile.NamedTemporaryFile(mode='w+') as f:
            yaml.dump(
                {
                    "model": {
                        "type": "base",
                        "generation": {
                            "multichoice_continuations_start_space": False,
                            "no_multichoice_continuations_start_space": False,
                        },
                        "base_params": {
                            "model_args": "pretrained=peft-internal-testing/tiny-OPTForCausalLM-lora",
                            "dtype": "float32",
                        },
                        "merged_weights": {
                            "base_model": "hf-internal-testing/tiny-random-OPTForCausalLM",
                            "adapter_weights": True,
                            "delta_weights": False,
                        }
                    }
                },
                f
            )
            f.seek(0)
            model_config = create_model_config(
                args=Namespace(model_args=None,
                               model_config_path=f.name,
                               override_batch_size=0,
                               use_chat_template=False),
                accelerator=None
            )
        env_config = EnvConfig(token=TOKEN, cache_dir=CACHE_PATH)
        model, _ = load_model(model_config, env_config)
        requests, docs = create_requests_from_tasks(
            task_dict=task_dict,
            fewshot_dict=fewshot_dict,
            num_fewshot_seeds=0,
            lm=model,
            max_samples=None,
            evaluation_tracker=evaluation_tracker,
            use_chat_template=False,
            system_prompt=None,
        )
        evaluate(
            model,requests, docs, task_dict, 0, evaluation_tracker
        )