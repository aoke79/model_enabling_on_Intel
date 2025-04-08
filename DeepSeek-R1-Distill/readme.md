this enabling bkm is for DeepSeek-R1-Distill models like qwen-1.5b, qwen-7b, llama-8b, qwen-14b based-on OpenVINO 2025.0.
please follow the steps:

1. download models from hf-mirror.com
    set HF_ENDPOINT=https://hf-mirror.com
    huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local-dir DeepSeek-R1-Distill-Llama-8B --local-dir-use-symlinks False

2. convert models by optimum-cli
    optimum-cli export openvino --model DeepSeek-R1-Distill-Qwen-1.5B --weight-format fp16 DeepSeek-R1-Distill-Qwen-1.5B-OV-FP16 --task text-generation-with-past --trust-remote-code  
  
3. setup python environment:
    conda create -n ds_qwen python=3.11
    conda activate ds_qwen
    pip install -r requirements.txxt
   
4. run models by the benchmark_genai.py like
    "python benchmark_genai.py --model "./../models/DeepSeek-R1-Distill-Qwen-1.5B-ov-g32-int4" --streamingflag 0 --type "qwen" --num_warmup 2 --num_iter 2 --max_new_tokens 100"

Good Luck
