# cd ./MedUniEval
# export HF_HOME="/mnt/hanxu/Xuhan-yzt/llm/data"
#export HF_ENDPOINT=https://hf-mirror.com
DATASETS_PATH="../data/Eval"
# EVAL_DATASETS="MedXpertQA-Text,MMLU-medical,EHRNoteQA,MMLU_Pro,MedMCQA,MedQA_USMLE,Medbullets_op4,Medbullets_op5,PubMedQA,SuperGPQA,MedXpertQA-MM,MMMU-Medical-test,MMMU-Medical-val"
# EVAL_DATASETS="MMLU_Pro,MedMCQA,MedQA_USMLE,Medbullets_op4,Medbullets_op5,PubMedQA,SuperGPQA"
# EVAL_DATASETS="PMC_VQA,VQA_RAD,SLAKE,PATH_VQA,Medmnist,EHRNoteQA,MMLU_Pro,MMLU-medical,MedXpertQA-Text,MedQA_USMLE,MedMCQA,PubMedQA,Medbullets_op4,Medbullets_op5,SuperGPQA,IU_XRAY,CheXpert_Plus"
EVAL_DATASETS="M3D"

MODEL_NAME="Hulumed_qwen2"
MODEL_PATH="../workspace/model/hulu_7B"
OUTPUT_PATH="../workspace/eval/hulu"
CONFIG_PATH="../config/config.json"
COMPRESSION="medpruner"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
USE_VLLM="False"
IFS=',' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPULIST[@]}
CHUNKS=8
#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1

# Eval LLM setting
MAX_NEW_TOKENS=256
MAX_IMAGE_NUM=600
TEMPERATURE=0
TOP_P=0.95
REPETITION_PENALTY=1.0


# LLM judge setting
USE_LLM_JUDGE="True"
GPT_MODEL="gpt-4.1-2025-04-14"
OPENAI_API_KEY="Your key"


for IDX in $(seq 0 $((CHUNKS-1))); do

    echo "Starting chunk $IDX in $CHUNKS"
    export chunk_idx=$IDX
    export num_chunks=$CHUNKS
    export MASTER_PORT=$((50000 + IDX))

    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --use_vllm "$USE_VLLM" \
    --compression "$COMPRESSION" \
    --config "$CONFIG_PATH" \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX \
    --reasoning $REASONING \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_gpt_model "$GPT_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --test_times "$TEST_TIMES"
    wait
done