from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
from deepeval.metrics import  AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset,Golden

answer_relevancy_metric = AnswerRelevancyMetric()

# Create goldens instead of test cases 

golden = Golden(
    
    input="who is prime minister of india",
    expected_output="Narendra Modi",
    context=["Narendra modi currently prime minister of india"],
    actual_output="Narendra Modi"
)

dataset = EvaluationDataset()
dataset.add_golden(golden)

for golden in dataset.goldens:
    test_case = LLMTestCase(
        input=golden.input,
        expected_output=golden.expected_output,
        retrieval_context=golden.context,
        actual_output=golden.actual_output
    )   

    dataset.add_test_case(test_case) 


evaluate(test_cases=dataset.test_cases,metrics=[AnswerRelevancyMetric()])

