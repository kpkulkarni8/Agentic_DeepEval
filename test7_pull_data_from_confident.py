from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
from deepeval.metrics import  AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset,Golden


test_data = [

        {
          "input":"who is prime minister of india",
          "output":"Narendra modi"
        },

        {
          "input":"who created python",
          "output":"van rossov"
        }
 
    ]

goldens = []

for data in test_data:
    golden = Golden(

    input=data['input'],
    expected_output=data['output']      
 )

goldens.append(golden)
#print(goldens)
eval_dataset = EvaluationDataset(goldens=goldens)
#pull data from confident ai 
eval_dataset = EvaluationDataset(goldens=goldens)
dataset_pulled = eval_dataset.pull(alias="TestGoldenDataTest")
print(dataset_pulled)


# for golden in dataset_pulled:
#     test_case = LLMTestCase(

#       input=golden.input,
#       expected_output=golden.output,
#       actual_output=
    