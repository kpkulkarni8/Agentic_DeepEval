from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from langchain_ollama import ChatOllama

# Local Ollama model for generating the answer
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen2.5-coder",
    temperature=0.5,
)

# Local Ollama model for judging the answer
judge_model = OllamaModel(
    model="qwen2.5-coder",
    base_url="http://localhost:11434",
    temperature=0,
)

answer_relevancy_metric = AnswerRelevancyMetric(
    model=judge_model,
    threshold=0.5,
    include_reason=True,
)

test_case = LLMTestCase(
    input="Who is the current president of the United States?",
    actual_output=llm.invoke(
        "Who is the current president of the United States?"
    ).content,
)

answer_relevancy_metric.measure(test_case)

print("Score:", answer_relevancy_metric.score)
print("Reason:", answer_relevancy_metric.reason)
print("Passed:", answer_relevancy_metric.is_successful())