import streamlit as st

def streamlit_UI(model):
    # 设置页面布局
    st.set_page_config(layout="wide")

    # 标题
    st.title("Streamlit 分屏布局示例")

    # 创建两列
    col1, col2 = st.columns(2)

    # 左边列：文本和按钮
    with col1:
        st.header("左侧内容")
        st.write("这里是左侧栏，显示文本和按钮。")
        st.markdown("**这是一个加粗的文本**，*这是一个斜体的文本*。")
        default_text = "点击这里输入内容"

        # 创建输入框并显示默认文本
        text_input = st.text_input(label='点击此处输入', value=default_text)
        
        # 按钮
        if st.button("点击我"):
            st.write(text_input)

    # 右边列：视频、图片和音频
    with col2:
        st.header("右侧内容")
        st.write("这里是右侧栏，显示视频、图片和音频。")
        
        # 显示本地图片
        st.subheader("1. 本地图片")
        image_path = "D:/MiniCPM-o26/test_samples/p2_0.png"  # 替换为你的本地图片路径
        st.image(image_path, caption="这是一张本地图片", use_column_width=True) #width = 600)
        
        # 显示本地视频
        st.subheader("2. 本地视频")
        video_path = "D:/MiniCPM-o26/test_samples/v0.mp4"  # 替换为你的本地视频路径
        st.video(video_path)
        
        # 显示本地音频
        st.subheader("3. 本地音频")
        audio_path = "D:/MiniCPM-o26/test_samples/林伊墨_摊破浣溪沙.mp3"  # 替换为你的本地音频路径
        st.audio(audio_path, format="audio/mp3")

    # 结束
    #st.write("感谢使用这个Streamlit示例程序！")

if __name__ == '__main__':
