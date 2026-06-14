## Quick Start

### VSCode Debug config: press F5 to launch

- .vscode/launch.json: corresponds to MedUniEval/eval_Amos.sh
- .vscode/launch_chunks.json: corresponds to MedUniEval/eval_M3D_chunked.sh

### Start Evaluation:

EVAL_DATASETS evaluation dataset


| Environment Variable  | Description                                              |
| -------------------- | -------------------------------------------------------- |
| EVAL_DATASETS        | Dataset type to evaluate (must match benchmarks.py)     |
| MODEL_NAME           | Model name (must match LLMs.py registration)              |
| MODEL_PATH           | Model path                                               |
| CUDA_VISIBLE_DEVICES | Visible CUDA devices                                     |
| USE_VLLM             | Whether to enable vllm                                   |
| CHUNKS               | Number of chunks                                         |

```bash
cd MedUniEval
# Evaluate Amos/3DRad
bash eval_Amos.sh
# Evaluate M3D
bash eval_M3D_chunked.sh
```
