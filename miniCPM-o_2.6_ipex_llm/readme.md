the link for intel ipex-llm is https://github.com/intel/ipex-llm/tree/main

to enabling it steps:
1. download models from hf-mirror.com

   set HF_ENDPOINT=https://hf-mirror.com
   huggingface-cli download --resume-download openbmb/MiniCPM-o-2_6 --local-dir MiniCPM-o-2_6 --local-dir-use-symlinks False

3. download the latest code from ipex-llm code base:

   git clone https://github.com/intel/ipex-llm.git

5. setup the environment

   under MiniCPM-o-2_6 folder of ipex-llm code base, like "ipex-llm-main\python\llm\example\GPU\HuggingFace\Multimodal\MiniCPM-o-2_6", you will get the readme.md and requrirements.txt file.

    however, I suggest going to Intel official ipex website to get the install-packages.
    https://intel.github.io/intel-extension-for-pytorch/
    https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.3.110%2Bxpu&os=windows&package=pip
    since minicpm-o need torch 2.3, so please choose v2.3.110+xpu version, and you can choose the install packages based-on both your hardware and cn/us (where you are).
   after installing the torch, torchaudio, torchvision and intel_extension_for_pytorch, you can install other packages, like "pip install -r requirements.txt".

7. run models like "python chat_images.py", or "streamlit run streamlit_demo.py"
    please note don't forget the change the model, image, video, audio path to your local folders.
