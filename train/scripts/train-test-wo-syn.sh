export WANDB_MODE=disabled

en_train_data="\  
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/ArguAna\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/COLIEE\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/ELI5\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/FEVER\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/FiQA\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/HotpotQA\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/MSMARCO\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/NLI_data\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/NQ\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/PubMedQA\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/Quora\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/SCIDOCS\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/SQuAD\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/STS_data\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/Trivia\
    "
zh_train_data="\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/zh/cMedQAv2\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/zh/DuReader\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/zh/Law_GPT\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/zh/mMARCO_zh\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/zh/Multi_CPR\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/zh/NLI_data\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/zh/STS_data\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/zh/T2Ranking\
    "
multilingual_train_data="\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/multilingual/MIRACL\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/multilingual/MrTyDi\
    "

synthetic_train_data="\
    /data/share/project/tr/mmteb/code/datasets/teccovid_generation_results/treccovid/formatted/11-13\
    /data/share/project/tr/mmteb/code/datasets/spartqa_generation_results/spartqa/formatted/11-14 \
    /data/share/project/psjin/data/generated_data/belebele/generation_results/hn_mine_data_sample \
    /data/share/project/tr/mmteb/code/datasets/winogrande_generation_results/winogrande/formatted/11-15\
    /data/share/project/psjin/data/generated_data/ailastatutes/generation_results/hn_mine_data_merged/en/ailastatutes \
    /data/share/project/psjin/data/generated_data/arguana/generation_results/hn_mine_data/en/arguana \
    /data/share/project/psjin/data/generated_data/covidretrieval/generation_results/hn_mine_data_cleaned/zh/covidretrieval\
    /data/share/project/psjin/data/generated_data/scidocs/generation_results/hn_mine_data/en/scidocs \
    "
train_data="\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/ArguAna\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/MSMARCO\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/en/SCIDOCS\
    /data/share/project/shared_datasets/bge-multilingual-gemma2-data/zh/mMARCO_zh\
"
    
# set large epochs and small batch size for testing
num_train_epochs=1
per_device_train_batch_size=32

num_gpus=4

export HF_HUB_CACHE="/data/share/project/shared_datasets/.cache"

model_args="\
    --model_name_or_path  /data/share/project/shared_models/Qwen3-8B-auto-eos\
    --cache_dir $HF_HUB_CACHE \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --save_merged_lora_model True \
"

data_args="\
    --train_data $train_data \
    --cache_path $HF_HUB_CACHE \
    --train_group_size 4 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_format 'Instruct: {}\nQuery: {}' \
    --knowledge_distillation True \
    --same_dataset_within_batch True \
    --small_threshold 0 \
    --drop_threshold 0 \
"

training_args="\
    --output_dir /data/share/project/psjin/model/geevec-qwen3-8b-v1-test-wo-syn \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --sub_batch_size 16 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed /data/share/project/public_envs/FlagEmbedding/examples/finetune/ds_stage1.json \
    --logging_steps 1 \
    --save_steps 1000 \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type kl_div \
    --max_example_num_per_dataset 50000 \
"


cmd="CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.embedder.decoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd