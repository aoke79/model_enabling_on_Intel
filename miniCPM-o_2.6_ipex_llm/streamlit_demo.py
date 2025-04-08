#from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoFeatureExtractor
import time
import librosa
import argparse

import streamlit as st
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoFeatureExtractor
from ipex_llm.transformers import AutoModel

# 设置页面布局
st.set_page_config(layout="wide")
#st.set_page_config(layout="centered")
# 初始化 session_state
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

#if "feature_extractor" not in st.session_state:
#    st.session_state.feature_extractor = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    #st.session_state.chat_history = [{
    #        "role": "system",
    #        "content": "You are a great AI assistant."
    #    }]
    
if "current_model_type" not in st.session_state:
    st.session_state.current_model_type = []

if "image" not in st.session_state:
    st.session_state.image = None

if "audio" not in st.session_state:
    st.session_state.audio = None

if "image_path" not in st.session_state:
    st.session_state.image_path = []

if "audio_path" not in st.session_state:
    st.session_state.audio_path = []

#if "once_content" not in st.session_state:
#    st.session_state.once_content = []

model_path = "D:/minicpm/model_MiniCPM_o26"
# 加载模型和分词器
def load_model(model_type):
    if model_type == "text":
        if st.session_state.current_model_type != "text":
            #st.session_state.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            #st.session_state.model = AutoModelForCausalLM.from_pretrained("gpt2")
            st.session_state.model = AutoModel.from_pretrained(model_path, 
                                      load_in_low_bit="sym_int4",
                                      optimize_model=True,
                                      trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      use_cache=True,
                                      init_vision=False,
                                      init_audio=False,
                                      init_tts=False,
                                      modules_to_not_convert=[])
            st.session_state.model.half().to("xpu")
            st.session_state.tokenizer = AutoTokenizer.from_pretrained(
                         model_path, trust_remote_code=True)
            st.session_state.current_model_type = "text"
    elif model_type == "image":
        if st.session_state.current_model_type != "image":
            #st.session_state.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
            st.session_state.model = AutoModel.from_pretrained(model_path, 
                                      load_in_low_bit="sym_int4",
                                      optimize_model=True,
                                      trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      use_cache=True,
                                      init_vision=True,
                                      init_audio=False,
                                      init_tts=False,
                                      modules_to_not_convert=["vpm", "resampler"])
            st.session_state.model.half().to("xpu")
            st.session_state.tokenizer = AutoTokenizer.from_pretrained(
                         model_path, trust_remote_code=True)
            st.session_state.current_model_type = "image"
    elif model_type == "audio":
        if st.session_state.current_model_type != "audio":
            #st.session_state.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
            st.session_state.model = AutoModel.from_pretrained(model_path, 
                                      load_in_low_bit="sym_int4",
                                      optimize_model=True,
                                      trust_remote_code=True,
                                      attn_implementation='sdpa',
                                      use_cache=True,
                                      init_vision=False,
                                      init_audio=True,
                                      init_tts=False,
                                      modules_to_not_convert=["apm"])
            st.session_state.model.half().to("xpu")
            st.session_state.tokenizer = AutoTokenizer.from_pretrained(
                         model_path, trust_remote_code=True)
            st.session_state.current_model_type = "audio"
    st.write("Load model finished")

# 生成聊天回复
def generate_response(prompts):
    print("Prompt is ", prompts)
    print("state is ", st.session_state.current_model_type)
    with torch.inference_mode():
        start_t = time.time()
        if st.session_state.current_model_type == "image":
            response = st.session_state.model.chat(
                image=st.session_state.image_path,
                msgs=prompts,
                tokenizer=st.session_state.tokenizer,
                sampling=True,
                top_p=sidebar_top_p,
                temperature=sidebar_temperature)
                #max_new_tokens=1024,)
            torch.xpu.synchronize()
        elif st.session_state.current_model_type == "audio":
            response = st.session_state.model.chat(
                audio=st.session_state.audio_path,
                msgs=prompts,
                tokenizer=st.session_state.tokenizer,
                sampling=True,
                top_p=sidebar_top_p,
                temperature=sidebar_temperature)
                #max_new_tokens=1024,)
            torch.xpu.synchronize()
        else:
            response ='Error'
        end_t = time.time()
        st.write("inference time is ", end_t - start_t)
    return response

# 页面布局
col1, col2 = st.columns([2, 3])

# 左侧：聊天界面
with col1:
    st.header("Chatbot")

    sidebar_name = st.sidebar.title(model_path)
    sidebar_max_length = st.sidebar.slider("max_length", 0, 4096, 2048, step=2)
    sidebar_top_p = st.sidebar.slider("top_p", 0.0, 1.0, 0.8, step=0.01)
    sidebar_temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.7, step=0.01)

    # Clear chat history button
    buttonClean = st.sidebar.button("Clear chat history", key="clean")
    if buttonClean:
        st.session_state.chat_history = []
        st.session_state.audio_path = []
        user_input = []
        st.rerun()

    user_input = st.text_input("请输入您的消息：", key="user_input")
    if user_input:
        st.session_state.chat_history.append({'role':'user', 'content':f"{user_input}"})
        st.write(st.session_state.chat_history)
        # 生成回复
        #time_s = time.time()
        response = generate_response(st.session_state.chat_history)
        #time_e = time.time()
        #print("inference time is ", time_e - time_s)
        #st.write(time_e - time_s)

        #st.session_state.chat_history.append(f"您: {user_input}")
        #st.session_state.chat_history.append(f"Chatbot: {response}")
        st.session_state.chat_history.append({'role':'assistant', 'content':f"{response}"})

    # 显示聊天历史
    st.subheader("对话历史")
    for message in st.session_state.chat_history:
        st.write(message)

# 右侧：文件选择和显示
with col2:
    st.header("multimedia files choose")
    file_type = st.radio("file type", ["image", "video", "audio"])

    if file_type == "image":
        uploaded_file = st.file_uploader("upload image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state.image = Image.open(uploaded_file)
            st.image(st.session_state.image, caption="uploaded image", use_column_width=True)
            load_model("image")
            st.session_state.image_path = st.session_state.image.convert('RGB')
            #inputs = st.session_state.feature_extractor(
            #    images=image, return_tensors="pt").to(st.session_state.model.device)
            #st.session_state.chat_history.append({'role':'user', 'content': f"{st.session_state.image}"})
            st.write("load model and uploaded image ", uploaded_file)

    elif file_type == "video":
        uploaded_file = st.file_uploader("upload video", type=["mp4", "mov"])
        if uploaded_file is not None:
            video_bytes = uploaded_file.read()
            st.video(video_bytes)
            st.write("视频模型未实现，请选择其他类型。")

    elif file_type == "audio":
        uploaded_file = st.file_uploader("upload audio", type=["mp3", "m4a", "wav"])
        if uploaded_file is not None:
            #audio_bytes = uploaded_file.read()
            #st.audio(audio_bytes, format="audio/wav")
            audio_input, _ = librosa.load(uploaded_file, sr=16000, mono=True)
            print("audio file is ", uploaded_file)
            st.session_state.audio_path = audio_input
            #st.audio(uploaded_file, format="audio/wav/mp3")
            load_model("audio")
            #st.session_state.audio_path, _ = librosa.load(uploaded_file, sr=16000, mono=False)
            #st.session_state.chat_history.append({'role':'user', 'content': f"{audio_input}"})
            #st.session_state.file_path = uploaded_file
            #st.write("load model and uploaded audio ", uploaded_file)

# 运行 Streamlit 应用
if __name__ == "__main__":
    st.write("欢迎使用多模态聊天机器人！")