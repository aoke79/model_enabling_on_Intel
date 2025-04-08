#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import time
import torch
from PIL import Image
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModel

model_path = "D:/minicpm/model_MiniCPM_o26"

# Load model in 4 bit,
# which convert the relevant layers in the model into INT4 format
model = AutoModel.from_pretrained(
    model_path, 
    load_in_low_bit="sym_int4",
    optimize_model=True,
    trust_remote_code=True,
    attn_implementation='sdpa',
    use_cache=True,
    init_vision=True,
    init_audio=False,
    init_tts=False,
    modules_to_not_convert=["vpm", "resampler"])

model = model.half().to('xpu')

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True)

def gh_text():
    prompts =[
        "冬天能穿多少穿多少，夏天能穿多少穿多少，你知道这两句话的意思么？",
        "中国男足谁都踢不过，中国乒乓球谁都打不过，你知道这两句话的意思么？",
        "夫妻双方吵架，女方说：孩子要是像你就死定了，男方说，孩子要是不像我，你就死定了。你知道这句话的意思么？",
        "有人掉河里了，我是先吃面包，还是先吃巧克力",
        "用‘万事如意’写首七言绝句藏头诗"]

    for question in prompts:
        messages = [{'role': 'user', 'content': question}]
        print("messages are ", messages)

        with torch.inference_mode():
            # ipex_llm model needs a warmup, then inference time can be accurate
            st = time.time()
            response = model.chat(
                msgs=messages,
                tokenizer=tokenizer,
                sampling=True,
                #max_new_tokens=args.n_predict,
            )
            torch.xpu.synchronize()
            end = time.time()

        print(f'Inference time: {end-st} s')
        #print('-'*20, 'Input Image Path', '-'*20)
        #print(image_path)
        print('-'*20, 'Input Prompt', '-'*20)
        print(question)
        print('-'*20, 'Chat Output', '-'*20)
        print(response)

gh_text()