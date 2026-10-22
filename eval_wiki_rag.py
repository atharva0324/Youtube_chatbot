from datasets import Dataset
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase,LLMTestCaseParams
from deepeval.metrics import FaithfulnessMetric,AnswerRelevancyMetric,ContextualRelevancyMetric,ContextualPrecisionMetric,ContextualRecallMetric
from deepeval.metrics import GEval
from chatbot import ingest_video, ask, retrievers_by_video,chat_histories_by_video
import json

import os
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "90"
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "900"


with open("golden_core.json") as f:
    goldens_core = json.load(f)

with open("golden_wikipedia.json") as f:
    golden_wikipedia=json.load(f)


video_id="i_LwzRVP7bg"
test_cases_core=[]
test_cases_wiki=[]
model="gpt-4o"
ingest_video(video_id)

for g in golden_wikipedia:
    chat_histories_by_video[video_id]=[]
    
    output=ask(video_id,g['question'],return_context=True)
    test_cases_core.append(
        LLMTestCase(
            input=g['question'],
            
            actual_output=output['answer'],
            expected_output=g['expected_answer']
           

        
        )
    )

correctness = GEval(
    name="Correctness",
    criteria=(
        "Check whether the actual output is factually consistent with the expected output. "
        "Extra correct detail is fine. Penalize contradictions and missing key facts."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.7,
    model=model,
)
metrics=[
    
    AnswerRelevancyMetric(threshold=0.85,model=model,include_reason=False),
    correctness
    
]

evaluate(test_cases=test_cases_core,metrics=metrics)

