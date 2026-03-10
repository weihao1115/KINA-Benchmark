import json
import logging
from string import ascii_uppercase
from typing import Dict, List

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.dynamic_metrics import multilingual_extractive_match_metric
from lighteval.metrics.utils.extractive_match_utils import IndicesExtractionConfig
from lighteval.utils.language import Language
from lighteval.tasks.requests import Doc

logger = logging.getLogger(__name__)


# Module-level singleton for answer extraction
# Uses the same extraction logic as gpqa_instruct_pass_at_1_Xn internally
_EXTRACTION_METRIC = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    gold_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    pred_extraction_target=[IndicesExtractionConfig(prefix_for_extraction="NativeLetters")],
    precision=6,
)


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    parsed_questions = []

    for item in data:
        question_data = {
            'id': item['_id'],
            'batch_id': item['batchId'],
            'question': None,
            'options': {},
            'ground_truth': None,
            'category': None,
            'source': None,
            'materials': None,
        }

        for label in item['labels']:
            label_type = label['data']['hash']

            if label_type == 'GPQA_QUESTION':
                question_data['question'] = label['data']['value']
                question_data['ground_truth'] = label['data'].get('correctAnswer')

            elif label_type == 'GPQA_TYPE':
                question_data['category'] = label['data']['value']

            elif label_type == 'GPQA_SOURCE':
                question_data['source'] = label['data']['value']

            elif label_type == 'GPQA_MATERIAL':
                question_data['materials'] = label['data']['value']

            elif label_type in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
                question_data['options'][label_type] = {
                    'id': label['data']['id'],
                    'answer': label['data']['answer'],
                    'explanation': label['data'].get('explanation'),
                    'source': label['data'].get('source')
                }

        parsed_questions.append(question_data)

    return parsed_questions


prompt_template = """Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format ’Answer: $LETTER’ (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J. Think step by step before answering.
Question: 
{question}
Options:
{options}"""


def get_messages(doc):
    question = doc['question']
    options = doc['options']
    option_str = "\n".join([f"{letter}: {choice['answer'].strip()}" for letter, choice in options.items()])

    messages = [
        {"role": "user", "content": prompt_template.format(question=question, options=option_str)}
    ]
    return messages


GPQA_METRICS = {
    1: Metrics.gpqa_instruct_pass_at_1_1n.value,
    4: Metrics.gpqa_instruct_pass_at_1_4n.value,
    8: Metrics.gpqa_instruct_pass_at_1_8n.value,
}


def get_gpqa_metric(n_sampling: int):
    """Get the appropriate GPQA metric based on n_sampling."""
    if n_sampling in GPQA_METRICS:
        return GPQA_METRICS[n_sampling]
    raise ValueError(
        f"n_sampling={n_sampling} is not supported. "
        f"Supported values are: {list(GPQA_METRICS.keys())}."
    )


def judge_score(metadata_doc: Dict, responses: List[str]):
    """
    Judge the score of responses against the ground truth.

    This function performs two tasks:
    1. Compute pass@1 score using gpqa_metric.compute()
    2. Extract predictions from each response individually to get complete extracted_predictions

    Why extract individually?
    ─────────────────────────────────────────────────────────────────────────
    lighteval's gpqa_instruct_pass_at_1_Xn metric internally uses the PassAtK class.
    PassAtK.compute() loops over each response, calling sample_scoring_function for each.

    The problem: each call to sample_scoring_function overwrites doc.specific["extracted_predictions"],
    so only the last response's extraction result is retained.

    Call chain:
        PassAtK.compute()
          └─> for pred in predictions:
                └─> sample_scoring_function(pred, ref, doc)
                      └─> multilingual_extractive_match_metric.sample_level_fn([ref], [pred], doc)
                            └─> doc.specific["extracted_predictions"] = [current pred's extraction]
                                                                          ↑ overwritten each time!

    Therefore, we need to call the extraction logic separately for each response to get
    the complete extracted_predictions. Verified by testing that individual extraction
    results are identical to those produced inside compute's internal loop.
    ─────────────────────────────────────────────────────────────────────────

    Args:
        metadata_doc: Document metadata containing question, options, and ground_truth
        responses: List of model responses to evaluate

    Returns:
        dict with:
            - 'score': float, pass@1 score (0.0 to 1.0)
            - 'extracted_predictions': List[str], extracted answer letter for each response
    """
    question = metadata_doc["question"]
    options = metadata_doc["options"]
    ground_truth = metadata_doc["ground_truth"]

    n_sampling = len(responses)
    num_choices = len(options)
    choices = list(ascii_uppercase[:num_choices])
    correct_index = ascii_uppercase.index(ground_truth)

    # ========== Step 1: Compute pass@1 score using gpqa_metric ==========
    # Create Doc object for gpqa_metric.compute()
    doc_for_score = Doc(
        query=question,
        choices=choices,
        gold_index=correct_index,
        instruction=question,
    )

    gpqa_metric = get_gpqa_metric(n_sampling)
    result = gpqa_metric.compute(
        golds=[ground_truth],
        predictions=responses,
        formatted_doc=doc_for_score,
    )
    score = list(result.values())[0]

    # ========== Step 2: Extract predictions from each response individually ==========
    # Uses the module-level singleton _EXTRACTION_METRIC, same extraction logic as gpqa_metric
    # Return format: [[preds_for_resp1], [preds_for_resp2], ...], each sublist corresponds to one response
    extracted_predictions = []

    for resp in responses:
        # Create an independent Doc object for each response to avoid overwrite issues
        doc_for_extraction = Doc(
            query=question,
            choices=choices,
            gold_index=correct_index,
            instruction=question,
        )

        # Call sample_level_fn for extraction
        # Note: the golds argument does not affect extraction results, only score computation
        _EXTRACTION_METRIC.sample_level_fn([ground_truth], [resp], doc_for_extraction)

        # Get extraction results from doc.specific
        if doc_for_extraction.specific:
            preds = doc_for_extraction.specific.get("extracted_predictions", [])
            extracted_predictions.append(preds)
        else:
            extracted_predictions.append([])

    return dict(score=score, extracted_predictions=extracted_predictions)


if __name__ == '__main__':
    data = load_data('data/SGPQA-Diamond.json')
    for doc in data[:3]:
        messages = get_messages(doc)
        print(messages)
