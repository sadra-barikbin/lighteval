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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from lighteval.logging.hierarchical_logger import hlog
from lighteval.models.adapter_model import AdapterModel
from lighteval.models.base_model import BaseModel
from lighteval.models.delta_model import DeltaModel
from lighteval.models.dummy_model import DummyModel
from lighteval.models.endpoints import (
    EndpointModel,
    InferenceEndpointModel,
    AnthropicModel,
    OpenAIModel,
)
from lighteval.models import model_config
from lighteval.models.model_config import (
    AdapterModelConfig,
    BaseModelConfig,
    DeltaModelConfig,
    DummyModelConfig,
    EnvConfig,
    InferenceEndpointModelConfig,
    InferenceModelConfig,
    TGIModelConfig,
    EndpointConfig,
)
from lighteval.models.endpoints.tgi_model import ModelClient
from lighteval.utils import NO_TGI_ERROR_MSG, is_accelerate_available, is_tgi_available


if is_accelerate_available():
    from accelerate.utils import calculate_maximum_sizes, convert_bytes


@dataclass
class ModelInfo:
    model_name: str
    model_sha: Optional[str] = None
    model_dtype: Optional[str] = None
    model_size: Optional[str] = None


def load_model(  # noqa: C901
    config: Union[
        BaseModelConfig,
        AdapterModelConfig,
        DeltaModelConfig,
        TGIModelConfig,
        InferenceEndpointModelConfig,
        DummyModelConfig,
        EndpointConfig,
    ],
    env_config: EnvConfig,
) -> Tuple[Union[BaseModel, AdapterModel, DeltaModel, ModelClient, DummyModel], ModelInfo]:
    """Will load either a model from an inference server or a model from a checkpoint, depending
    on the config type.

    Args:
        args (Namespace): arguments passed to the program
        accelerator (Accelerator): Accelerator that will be used by the model

    Raises:
        ValueError: If you try to load a model from an inference server and from a checkpoint at the same time
        ValueError: If you try to have both the multichoice continuations start with a space and not to start with a space
        ValueError: If you did not specify a base model when using delta weights or adapter weights

    Returns:
        Union[BaseModel, AdapterModel, DeltaModel, ModelClient]: The model that will be evaluated
    """
    # Isn't better for each config to bear the respnsibility for loading its model itself?
    match type(config):
        # Inference server loading
        case model_config.TGIModelConfig:
            return load_model_with_tgi(config)

        case model_config.InferenceEndpointModelConfig | model_config.InferenceModelConfig:
            return load_model_with_inference_endpoints(config, env_config=env_config)

        case model_config.BaseModelConfig:
            return load_model_with_accelerate_or_default(config=config, env_config=env_config)
    
        case model_config.EndpointConfig:
            return load_model_with_endpoint(config=config)

        case model_config.DummyModelConfig:
            return load_dummy_model(config=config, env_config=env_config)


def load_model_with_tgi(config: TGIModelConfig):
    if not is_tgi_available():
        raise ImportError(NO_TGI_ERROR_MSG)

    hlog(f"Load model from inference server: {config.inference_server_address}")
    model = ModelClient(
        address=config.inference_server_address, auth_token=config.inference_server_auth, model_id=config.model_id
    )
    model_name = str(model.model_info["model_id"])
    model_sha = model.model_info["model_sha"]
    model_precision = model.model_info["model_dtype"]
    model_size = -1
    model_info = ModelInfo(
        model_name=model_name,
        model_sha=model_sha,
        model_dtype=model_precision,
        model_size=model_size,
    )
    return model, model_info


def load_model_with_inference_endpoints(config: InferenceEndpointModelConfig, env_config: EnvConfig):
    hlog("Spin up model using inference endpoint.")
    model = InferenceEndpointModel(config=config, env_config=env_config)
    model_info = ModelInfo(
        model_name=model.name,
        model_sha=model.revision,
        model_dtype=config.model_dtype or "default",
        model_size=-1,
    )
    return model, model_info


def load_model_with_accelerate_or_default(
    config: Union[AdapterModelConfig, BaseModelConfig, DeltaModelConfig], env_config: EnvConfig
):
    if isinstance(config, AdapterModelConfig):
        model = AdapterModel(config=config, env_config=env_config)
    elif isinstance(config, DeltaModelConfig):
        model = DeltaModel(config=config, env_config=env_config)
    else:
        model = BaseModel(config=config, env_config=env_config)

    model_name = model.model_name
    model_sha = model.model_sha
    model_precision = str(model.precision)
    if is_accelerate_available():
        model_size, _ = calculate_maximum_sizes(model.model)
        model_size = convert_bytes(model_size)
    else:
        model_size = -1
    model_info = ModelInfo(
        model_name=model_name,
        model_sha=model_sha,
        model_dtype=model_precision,
        model_size=model_size,
    )
    hlog(f"Model info: {model_info}")

    return model, model_info


def load_model_with_endpoint(config: EndpointConfig) -> EndpointModel:
    match config.type:
        case "anthropic":
            return AnthropicModel(config.model_id), ModelInfo(f"anthropic-{config.model_id}")
        case "openai":
            return OpenAIModel(config.model_id), ModelInfo(f"openai-{config.model_id}")


def load_dummy_model(config: DummyModelConfig, env_config: EnvConfig):
    return DummyModel(config=config, env_config=env_config), ModelInfo(model_name="dummy", model_sha=str(config.seed))
