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


def compare_two_images():
    # The following code for generation is adapted from 
    # https://huggingface.co/openbmb/MiniCPM-o-2_6#addressing-various-audio-understanding-tasks and
    # https://huggingface.co/openbmb/MiniCPM-o-2_6#chat-with-single-image
    image_path_1 = "D:/MiniCPM/datasets/mahuateng.jpg"
    image_path_2 = "D:/MiniCPM/datasets/0727_r.png"
    image_input_1 = Image.open(image_path_1).convert('RGB')
    image_input_2 = Image.open(image_path_2).convert('RGB')
    #question = 'Compare image 1 and image 2, tell me about the differences between image 1 and image 2.'
    question = '比较图1和图2，图2中的人是否在图1中？'

    messages = [{'role': 'user', 'content': [image_input_1, image_input_2, question]}]

    print("model_path is ", model_path)
    print("image_path is ", image_input_1)
    print("audio_path is ", image_input_2)
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

def gh_image():
    # The following code for generation is adapted from 
    image_path = "D:/MiniCPM/datasets/test_image/2_人物.png"
    image_input = Image.open(image_path).convert('RGB')
    question = '照片中的人是谁？'

    messages = [{'role': 'user', 'content': [image_input, question]}]

    print("model_path is ", model_path)
    print("image_path is ", image_input)
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

gh_image()