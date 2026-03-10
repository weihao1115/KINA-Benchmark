# KINA-Benchmark

## Environment Setup

```bash
conda create -n kina_bench python=3.10 -y && conda activate kina_bench
pip install -e .
```

### SGLang (Optional)

If you need to deploy local models using SGLang, install it separately:

```bash
pip install sglang[all]
```

Refer to the [SGLang documentation](https://github.com/sgl-project/sglang) for detailed installation instructions.

## Usage

### Method 1: Commercial API

Use commercial API endpoints directly (OpenAI, OpenRouter, etc.):

```bash
# Set environment variables
export OPENAI_BASE="https://api.openai.com/v1"        # or https://openrouter.ai/api/v1
export OPENAI_KEY="your-api-key"

# Run evaluation
python src/kina_bench/run_openai_chat.py \
    --model_id "gpt-4o" \
    --data_name "KINA-899" \
    --n_thread 32
```

You can also use a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
# Edit .env with your API credentials
```

### Method 2: Local Model with SGLang

Deploy a local model server using SGLang, then run evaluation:

**Step 1: Start SGLang Server**

```bash
# Basic usage
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --port 8000

# With thinking/reasoning mode (for Qwen3 series)
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --port 8000 \
    --reasoning-parser qwen3

# Multi-GPU with tensor parallelism
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-30B-A3B \
    --port 8000 \
    --tp 4
```

**Step 2: Run Evaluation**

```bash
export OPENAI_BASE="http://localhost:8000/v1"
export OPENAI_KEY="EMPTY"

python src/kina_bench/run_openai_chat.py \
    --model_id "Qwen/Qwen3-8B" \
    --data_name "KINA-899" \
    --n_thread 64
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | (required) | Model identifier |
| `--data_name` | `KINA-899` | Dataset name (without `.json` extension) |
| `--n_sampling` | `1` | Number of samples per question (1, 4, or 8) |
| `--max_tokens` | `16384` | Maximum output tokens |
| `--think_mode` | `none` | Thinking mode: `none`, `think`, or `nothink` |
| `--reasoning_effort` | `None` | Reasoning effort for o1/gpt-5 series: `low`, `medium`, `high` |
| `--n_thread` | `1` | Number of concurrent threads |
| `--overwrite` | `False` | Overwrite existing results |
| `--timeout` | `300` | Timeout in seconds for each API request |
| `--limit` | `None` | Limit total number of data samples for debugging |
| `--skip_inference` | `False` | Skip inference, only run evaluation (see note below) |

> **Note on `--skip_inference`**: If some samples fail during inference, the evaluation stage will not run by default. You need to re-run the same command to resume and retry the failed samples. If you want to evaluate the successfully completed samples first, add `--skip_inference` to your command.

### View Results

```bash
python src/kina_bench/pretty_print.py --data_name KINA-899
```

## Data Format

### Input: `data/{data_name}.json`

The input data is exported from a labeling platform with the following structure:

```json
[
    {
        "_id": "68da951b00128015648efdc9",
        "batchId": "68da951b00128015648efdc4",
        "labels": [
            {
                "data": {
                    "hash": "GPQA_QUESTION",
                    "value": "Which of the following statements...",
                    "correctAnswer": "C"
                }
            },
            {
                "data": {
                    "hash": "GPQA_TYPE",
                    "value": "Sociology/Sociology/Social and Folklore Studies"
                }
            },
            {
                "data": {
                    "hash": "A",
                    "answer": "Option A content...",
                    "explanation": "Explanation for option A..."
                }
            },
            {
                "data": {
                    "hash": "B",
                    "answer": "Option B content...",
                    "explanation": "..."
                }
            }
        ]
    }
]
```

Key fields:
- `GPQA_QUESTION`: Question text and correct answer
- `GPQA_TYPE`: Category/domain of the question
- `A`, `B`, `C`, ...: Answer options (up to J)

### Output: `results/{model_name}/n{n_sampling}_tokens{max_tokens}/`

**Intermediate file: `{data_name}.jsonl`**

Each line is a JSON object containing raw inference results:

```json
{
    "id": "68da951b00128015648efdec",
    "request": "[{\"role\": \"user\", \"content\": \"Answer the following...\"}]",
    "responses": [
        {"content": "Let me analyze...\n\nAnswer: A", "reasoning": null},
        {"content": "After careful thought...\n\nAnswer: A", "reasoning": null}
    ],
    "metadata": {
        "id": "...",
        "question": "...",
        "options": {"A": {...}, "B": {...}},
        "ground_truth": "A"
    },
    "total_input_tokens": 1234,
    "total_output_tokens": 567
}
```

**Final file: `{data_name}.json`**

Evaluation results with scores:

```json
[
    {
        "id": "68da951b00128015648efdec",
        "score": 0.75,
        "extracted_predictions": [
            ["A", "A"],
            ["A", "A"],
            ["A", "A"],
            ["H", "H"]
        ],
        "gt": "A"
    }
]
```

Fields:
- `score`: pass@1 score (0.0 to 1.0)
- `extracted_predictions`: Extracted answers for each response (list of lists)
- `gt`: Ground truth answer

## Directory Structure

```
kina_bench/
├── data/
│   └── KINA-899.json              # Input dataset
├── results/
│   └── {model_name}/
│       └── n{n}_tokens{max}/
│           ├── {data_name}.jsonl     # Raw inference results
│           └── {data_name}.json      # Evaluation scores
├── src/
│   └── kina_bench/
│       ├── run_openai_chat.py        # Main inference script
│       ├── utils.py                  # Data loading and scoring
│       ├── pretty_print.py           # Results visualization
│       └── config.py                 # Project configuration
└── pyproject.toml
```
