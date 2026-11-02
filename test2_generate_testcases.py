from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

context_precision_metrics = ContextualPrecisionMetric()

test_case = LLMTestCase(

    input="Who is current prime minister of india" ,   # input
    actual_output="Narendra Modi", # output should come LLM , Agent or RAG
    expected_output="Narendra modi is prime minister of india",
    retrieval_context=["Narendra is current prime minister"] #expected output 

)

context_precision_metrics.measure(test_case=test_case)
print("Score:", context_precision_metrics.score)
print("Success:", context_precision_metrics.success)
print("Reason:", context_precision_metrics.reason)