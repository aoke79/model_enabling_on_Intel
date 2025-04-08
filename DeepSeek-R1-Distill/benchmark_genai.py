# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai as ov_genai

def streamer(subword):
    print(subword, end='', flush=True)
    #print(subword, end='', flush=False)
    # Return flag corresponds whether generation should be stopped.
    # False means continue generation.
    return False

def setup_config(max_tokens, model_type="qwen"):
    config = ov_genai.GenerationConfig()
    
    if model_type == "qwen":
        config.eos_token_id = 151643
        config.stop_token_ids = {151643, 151647}
    elif model_type == "llama":
        config.eos_token_id = 128001
        config.stop_token_ids = {128001}

    config.do_sample = False
    config.repetition_penalty = 1.1
    config.top_k = 50
    config.top_p = 0.95
    config.max_new_tokens = max_tokens
    return config

def metrics_out(perf_metrics):
    print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
    print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
    print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
    print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
    
    print(f"in token number: {perf_metrics.get_num_input_tokens()}") 
    print(f"ou token number: {perf_metrics.get_num_generated_tokens()}") 
    
    print(f"Time per Output Token (TPOT): {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms/token")
    print(f"Time To the First Token (TTFT): {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
    print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")
    print("\n\n")

def main():
    global history
    parser = argparse.ArgumentParser(description="Help command")
    parser.add_argument(
        "-m", "--model", type=str,
        default="./../models/DeepSeek-R1-Distill-Qwen-1.5B-OV-INT4",
        help="Model folder including DeepSeek-R1 Distill OpenVINO Models")
    parser.add_argument(
        "-p", "--prompt", type=str, default="The Sky is blue because", help="Prompt")
    parser.add_argument(
        "-f", "--prompt_file", type=str, 
        default="./LLM_Test_Prompt_1k.txt", help="input prompt file")
    parser.add_argument(
        "-nw", "--num_warmup", type=int, default=1, help="Number of warmup iterations")
    parser.add_argument(
        "-n", "--num_iter", type=int, default=1, help="Number of iterations")
    parser.add_argument(
        "-mt", "--max_new_tokens", type=int, default=1000, help="Maximal number of new tokens")
    parser.add_argument(
        "-d", "--device", type=str, default="GPU", help="Device")
    parser.add_argument(
        "-s", "--streamingflag", type=int, default=1, help="Streaming or not?")
    parser.add_argument(
        "-k", "--type", type=str, default="qwen", help="Streaming or not?")
    
    args = parser.parse_args()

    # Perf metrics is stored in DecodedResults. 
    # In order to get DecodedResults instead of a string input should be a list.
    #prompt = [args.prompt]
    models_path = args.model
    device = args.device
    num_warmup = args.num_warmup
    num_iter = args.num_iter
    file_path = args.prompt_file
    streaming_flag = args.streamingflag
    print("num_warmup is ", args.num_warmup)
    print("num_iter is ", args.num_iter)
    print("streaming_flag  is ", streaming_flag)
    print("max output len is ", args.max_new_tokens)
    print("type is ", args.type)
    
    cache_dir ="./model_cache"
    ov_setup = {"CACHE_DIR": cache_dir,
                 "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0",
                 "KV_CACHE_PRECISION": "f16"}
    
    pipe = ov_genai.LLMPipeline(models_path, device, **ov_setup)
    #prompt = [open(f"./LLM_Test_Prompt_1k.txt", 'r', encoding='utf-8').read()]
    prompt = [open(file_path, 'r', encoding='utf-8').read()]
    print("input prompt len is ", len(prompt[0]))
    #prompt = open(file_path, 'r', encoding='utf-8').read()
    #print("input prompt len is ", len(prompt))
    ov_config = setup_config(args.max_new_tokens, args.type)
    
    if streaming_flag == 1:
        res = pipe.generate(prompt, ov_config, streamer)
        perf_metrics = res.perf_metrics
        #print("\n")
        #print("warm up 1\n")
        for _ in range(num_warmup-1):
            res = pipe.generate(prompt, ov_config, streamer)
            perf_metrics += res.perf_metrics
        print("\n")
        print(f"Warm up {num_warmup} round")
        metrics_out(perf_metrics)
        
        res = pipe.generate(prompt, ov_config, streamer)
        perf_metrics = res.perf_metrics
        #print("\n")
        #print("iter 1")
        for _ in range(num_iter-1):
            res = pipe.generate(prompt, ov_config, streamer)
            perf_metrics += res.perf_metrics
        print("\n")
        print(f"iter {num_iter} round")
        metrics_out(perf_metrics)
    else:
        res = pipe.generate(prompt, ov_config)
        perf_metrics = res.perf_metrics
        for _ in range(num_warmup-1):
            res = pipe.generate(prompt, ov_config)
            perf_metrics += res.perf_metrics
        print("\n")
        print(f"Warm up {num_warmup} round")
        metrics_out(perf_metrics)

        res = pipe.generate(prompt, ov_config)
        perf_metrics = res.perf_metrics
        for _ in range(num_iter-1):
            res = pipe.generate(prompt, ov_config)
            perf_metrics += res.perf_metrics
        print("\n")
        print(f"iter {num_iter} round")
        metrics_out(perf_metrics)

    '''
    for _ in range(num_iter - 1):
        print("num_iter is ", num_iter)
        res = pipe.generate(prompt, ov_config, streamer)
        perf_metrics += res.perf_metrics
        print("Round {num_iter}:")
        print(f"Load time: {perf_metrics.get_load_time():.2f} ms")
        print(f"Generate time: {perf_metrics.get_generate_duration().mean:.2f} ± {perf_metrics.get_generate_duration().std:.2f} ms")
        print(f"Tokenization time: {perf_metrics.get_tokenization_duration().mean:.2f} ± {perf_metrics.get_tokenization_duration().std:.2f} ms")
        print(f"Detokenization time: {perf_metrics.get_detokenization_duration().mean:.2f} ± {perf_metrics.get_detokenization_duration().std:.2f} ms")
        
        print(f"in token number: {perf_metrics.get_num_input_tokens()}") 
        print(f"ou token number: {perf_metrics.get_num_generated_tokens()}") 
        
        print(f"Time per Output Token (TPOT): {perf_metrics.get_tpot().mean:.2f} ± {perf_metrics.get_tpot().std:.2f} ms/token")
        print(f"Time To the First Token (TTFT): {perf_metrics.get_ttft().mean:.2f} ± {perf_metrics.get_ttft().std:.2f} ms")
        print(f"Throughput : {perf_metrics.get_throughput().mean:.2f} ± {perf_metrics.get_throughput().std:.2f} tokens/s")
        print("\n\n")
    '''

if __name__ == "__main__":
    main()
