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

import random
import re
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, Any

import torch


@dataclass
class ModelReturn:
    result: Union[tuple, list, str]
    input_tokens: list[int] = field(default_factory=list)  # model inputs
    generated_tokens: list[int] = field(default_factory=list)  # model generations
    truncated_tokens_count: Optional[int] = None  # How many tokens truncated
    padded_tokens_count: Optional[int] = None  # How many tokens of padding

    def get_result_for_eval(self):
        raise NotImplementedError()


@dataclass
class LoglikelihoodReturn(ModelReturn):
    # Float: Total log prob of the continuation
    # Optional(Bool): Whether the continuation is greedy (= all the tokens in the continuation are argmax of prob)
    result: Union[tuple[float, bool], float] = field(default_factory=tuple[float, bool])

    def get_result_for_eval(self):
        return self.result


@dataclass
class LoglikelihoodSingleTokenReturn(ModelReturn):
    # Log probs of the various single token options
    result: list[float] = field(default_factory=list)

    def get_result_for_eval(self):
        return self.result


@dataclass
class GenerateReturn(ModelReturn):
    result: str = field(default_factory=str)  # generated text continuation
    #This shouldn't be logprobs?
    logits: Optional[list[float]] = None  # Generated text logits

    def get_result_for_eval(self):
        return self.result if self.logits is None else (self.result, self.logits)


@dataclass
class GenerateMultiTurnReturn(ModelReturn):
    result: list[str] = field(default_factory=list)

    def get_result_for_eval(self):
        return self.result


@dataclass
class Batch:
    input_ids: torch.Tensor
    input_mask: torch.Tensor
    input_lengths: list[int]
    truncated: list[int]
    padded: list[int]


class AnswerExtractor:
    @abstractmethod
    def __call__(self, result: str):
        ...
    
    @abstractmethod
    def as_dict(self) -> dict:
        ...
    
    # Bad hack. Thanks to LightevalTaskConfig's becoming dict in the beginning of the evaluation!
    # Maybe it's fixed now in main branch.
    @classmethod
    def from_dict(cls, extractor_dict: dict[str, Any]) -> "AnswerExtractor":
        match extractor_dict["type"]:
            case "regex":
                return RegexAnswerExtractor(extractor_dict["regex_list"], extractor_dict["fallback"])
            case "separator":
                return SeparatorAnswerExtractor(extractor_dict["separator"])
            case _:
                raise ValueError("Unknown answer extractor type")


class RegexAnswerExtractor(AnswerExtractor):
    def __init__(self, regex_list: list[re.Pattern|str], fallback: int|Literal["random", "keep"] = "keep"):
        self.regex_list: list[re.Pattern] = list(map(re.compile, regex_list))
        self.fallback = fallback
    
    def __call__(self, result: str, choices: list[str]) -> str:
        for pattern in self.regex_list:
            choice = next(iter(re.findall(pattern, result)), "")
            if choice in choices:
                return choice
        if self.fallback == "random":
            return random.choice(choices)
        elif self.fallback == "keep":
            return result
        else:
            return choices[self.fallback]
    
    def as_dict(self) -> dict:
        return {
            "type": "regex",
            "regex_list": [p.pattern for p in self.regex_list],
            "fallback": self.fallback
        }


class SeparatorAnswerExtractor(AnswerExtractor):
    def __init__(self, separator_pattern: str|re.Pattern):
        self.separator = re.compile(separator_pattern)

    def __call__(self, result: str, choices: list[str]) -> list[str]:
        return [item.strip() for item in re.split(self.separator, result)]
    
    def as_dict(self):
        return {
            "type": "separator",
            "separator": self.separator.pattern
        }