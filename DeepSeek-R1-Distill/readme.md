this enabling bkm is for DeepSeek-R1-Distill models like qwen-1.5b, qwen-7b, llama-8b, qwen-14b based-on OpenVINO 2025.0.
please follow the steps:
1. download models from hf-mirror.com
2. convert models
3. run models by the benchmark_genai.py like "python benchmark_genai.py --model "./../models/DeepSeek-R1-Distill-Qwen-1.5B-ov-g32-int4" --streamingflag 0 --type "qwen" --num_warmup 2 --num_iter 2 --max_new_tokens 100"

Good Luck
