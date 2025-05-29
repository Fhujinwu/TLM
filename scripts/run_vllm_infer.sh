python scripts/vllm_infer.py --model_name_or_path /hujinwu/LLM_Assemble/pretrain_model/Meta-Llama-3-8B-Instruct \
    --adapter_name_or_path  /hujinwu/wyf/projects/zhangzitian/projects/TLM/saves/llama3-8b/offline_ttl/geosignal_5k/threshold_3-lr_5e-5-seed_42-v2 \
    --template llama3 \
    --dataset geosignal_5k \
    --output_dir $adapter_name_or_path \