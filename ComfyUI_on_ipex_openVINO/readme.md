there are two Intel SW Stack support ComfyUI running on Intel GPU, Ipex and OpenVINO, there are the steps:

by Ipex Xpu:
https://github.com/comfyanonymous/ComfyUI
https://pytorch.org/docs/main/notes/get_start_xpu.html
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

By openVINO:
pip install --pre torch torchvision torchaudio
pip install openvino
python main.py --cpu --use-pytorch-cross-attention
Since currently, the code was not uploaded, so please refer the link: https://github.com/comfyanonymous/ComfyUI/pull/6638 to add one note to enable it.

