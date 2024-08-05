import docker
import os
import pytest
import random
import requests
import time
from typing import Generator

from lighteval.models.model_config import EnvConfig
from lighteval.models import endpoints
from lighteval.models.endpoints import (
    EndpointModel,
    AnthropicModel,
    TGIModel,
    OpenAIModel,
    InferenceEndpointModel,
)


TOKEN = os.environ.get("HF_TOKEN")
CACHE_PATH = os.getenv("HF_HOME", ".")

@pytest.fixture(scope='package')
def tgi_model() -> Generator[TGIModel, None, None]:
    client = docker.from_env()
    port = random.randint(8000, 9000)
    container = client.containers.run(
        "ghcr.io/huggingface/text-generation-inference:2.2.0",
        command=[
            "--model-id", "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "--dtype", "float16",
        ],
        detach=True,
        name="lighteval-tgi-model-test",
        auto_remove=True,
        ports={"80/tcp": port}
    )
    address = f"http://localhost:{port}"
    for _ in range(30):
        try:
            if requests.get(f"{address}/health"):
                break
        except:
            time.sleep(1)
    else:
        raise RuntimeError("Couldn't setup TGI server.")
    model = TGIModel(address)
    yield model
    container.stop()
    container.wait()
    model.cleanup()


@pytest.fixture(scope='package')
def anthropic_model():
    model =  AnthropicModel("claude-3-5-sonnet-20240620")
    yield model
    model.cleanup()


@pytest.fixture(scope='package')
def openai_model():
    model =  OpenAIModel("davinci-002")
    yield model
    model.cleanup()


@pytest.fixture(scope='package')
def endpoint_model(request, tgi_model, anthropic_model, openai_model) -> Generator[EndpointModel, None, None]:
    match request.param:
        case endpoints.AnthropicModel:
            return anthropic_model
        case endpoints.TGIModel:
            return tgi_model
        case endpoints.OpenAIModel:
            return openai_model