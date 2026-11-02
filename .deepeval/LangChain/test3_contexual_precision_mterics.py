from deepeval.metrics import ContextualPrecisionMetric
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

contexual_precision_metric = ContextualPrecisionMetric(
    model=judge_model,
    threshold=0.5,
    include_reason=True,
)


test_case = LLMTestCase(
    input="What are types of bias can LLM generate , give me heading",
    actual_output=llm.invoke(
        "What are types of bias can LLM generate , give me heading"
    ).content,
    retrieval_context=["Gender Bia , Ethnic Bias , Racial Bias , Political Bias"],
    expected_output=""" **Gender Bias** **Reliogous Bias** **ethnic Bias** """
)

contexual_precision_metric.measure(test_case)

print("Score:", contexual_precision_metric.score)
print("Reason:", contexual_precision_metric.reason)
print("Passed:", contexual_precision_metric.is_successful())

