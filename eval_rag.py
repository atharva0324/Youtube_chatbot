from datasets import Dataset
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.test_case import LLMTestCase
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

for g in goldens_core:
    chat_histories_by_video[video_id]=[]
    retriever=retrievers_by_video[video_id]
    docs=retriever.invoke(g['question'])
    output=ask(video_id,g['question'],return_context=True)
    test_cases_core.append(
        LLMTestCase(
            input=g['question'],
            retrieval_context=output['context'],
            actual_output=output['answer'],
            expected_output=g['expected_answer']

        
        )
    )

metrics=[
    FaithfulnessMetric(threshold=0.85,model=model,include_reason=False),
    AnswerRelevancyMetric(threshold=0.85,model=model,include_reason=False),
    ContextualRecallMetric(threshold=0.7,model=model,include_reason=False),
    ContextualPrecisionMetric(threshold=0.7,model=model,include_reason=False)
]

evaluate(test_cases=test_cases_core,metrics=metrics)

