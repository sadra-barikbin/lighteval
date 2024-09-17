import pytest

from lighteval.models.endpoints import AnthropicModel
from lighteval.tasks.requests import GreedyUntilRequest


@pytest.fixture(scope='module')
def anthropic_model():
    model =  AnthropicModel("claude-3-5-sonnet-20240620")
    yield model
    model.cleanup()


def test_anthropic_model_api(anthropic_model: AnthropicModel):
    requests = [
        GreedyUntilRequest("test_task", 0, 0, "How many hands does human have?", [], 5, num_samples=1),
        GreedyUntilRequest("test_task", 0, 0, "How many eyes does human have?", [], 5, num_samples=1)
    ]
    returns = anthropic_model.greedy_until(requests)
    assert len(returns) == 2
    assert all((type(r.result) is str) and len(r.result) for r in returns)

    with pytest.raises(ValueError, match=r"Anthropic models work only with generative metrics"):
        anthropic_model.loglikelihood([])

    anthropic_model.tok_encode("Hi there")

    with pytest.raises(ValueError, match=r"Asking to pad but the tokenizer does not have a padding token."):
        anthropic_model.tok_encode(["Hi there!", "Hi!"])