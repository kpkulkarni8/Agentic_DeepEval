from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
from deepeval.metrics import  AnswerRelevancyMetric

answer_relevancy_metric = AnswerRelevancyMetric()


test_case1 = LLMTestCase(

    input="Who is current prime minister of india" ,   # input
    actual_output="Narendra Modi", # output should come LLM , Agent or RAG
    expected_output="Narendra modi is prime minister of india",
    retrieval_context=["Narendra is current prime minister"] #expected output 

)

test_case2 = LLMTestCase(

    input="Who buit taj mahal" ,   # input
    actual_output="Doctor", # output should come LLM , Agent or RAG
    expected_output="Narendra modi",
    retrieval_context=["Shahajhan built tajmahal"]
)

evaluate(test_cases=[test_case1,test_case2],metrics=[answer_relevancy_metric])