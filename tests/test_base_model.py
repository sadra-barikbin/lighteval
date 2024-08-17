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

from typing import Tuple, cast

from lighteval.models.base_model import BaseModel
from lighteval.models.model_config import BaseModelConfig, EnvConfig
from lighteval.models.model_loader import ModelInfo, load_model


def test_empty_requests():
    model_config = BaseModelConfig("trl-internal-testing/tiny-random-LlamaForCausalLM")
    model, _ = cast(Tuple[BaseModel, ModelInfo], load_model(config=model_config, env_config=EnvConfig()))

    assert model.loglikelihood([]) == []
    assert model.loglikelihood_single_token([]) == []
    assert model.loglikelihood_rolling([]) == []
    assert model.greedy_until([]) == []
    assert model.greedy_until_multi_turn([]) == []
