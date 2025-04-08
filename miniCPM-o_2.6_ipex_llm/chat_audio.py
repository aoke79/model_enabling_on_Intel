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
import librosa
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
    init_vision=False,
    init_audio=True,
    init_tts=False,
    modules_to_not_convert=["apm"])

model = model.half().to('xpu')
#model.init_tts()
#model.tts.float()

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True)

#audio_in_path_1 = "D:/MiniCPM/datasets/trump/meet_0.wav"
audio_in_path_1 = "D:/MiniCPM/datasets/lin/mo_ci1.wav"

#audio_ou_path_1 = "D:/MiniCPM/datasets/trump/output.wav"

#mimick_prompt = "Please repeat each user's speech, including voice style and speech content."
# prompt = "record every words spoken in this audio, and then translate them into Chinese."
#prompt = "record every words spoken in this audio"
#prompt = "说话的人是一个什么样的人？她的心情如何？"
#prompt = "请描述此音频的内容，如果有乐器，请说明是什么乐器？曲目是什么？"
#prompt = "请描述此音频的内容"
prompt = "请描述此音频的内容，如果是一首诗的话，请问出处，以及作者是谁？朗读者的心情如何？"
audio_input, _ = librosa.load(audio_in_path_1, sr=16000, mono=True) # load the audio to be mimicked

# `./assets/input_examples/fast-pace.wav`, 
# `./assets/input_examples/chi-english-1.wav` 
# `./assets/input_examples/exciting-emotion.wav` 
# for different aspects of speech-centric features.

messages = [{'role': 'user', 'content': [prompt, audio_input]}]
with torch.inference_mode():
    # ipex_llm model needs a warmup, then inference time can be accurate
    st = time.time()
    response = model.chat(
        msgs=messages,
        tokenizer=tokenizer,
        sampling=True,
        #max_new_tokens=1024,
        #generate_audio=True,
        #output_audio_path=audio_ou_path_1
    )
    torch.xpu.synchronize()
    end = time.time()

print(f'Inference time: {end-st} s')
print('-'*20, 'Input Audio Path', '-'*20)
print(audio_in_path_1)
#print('-'*20, 'Output Audio Path', '-'*20)
#print(audio_ou_path_1)
print('-'*20, 'Input Prompt', '-'*20)
print(prompt)
print('-'*20, 'Chat Output', '-'*20)
print(response)

