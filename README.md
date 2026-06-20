# llama.cpp

branch lastest kytensor-llm

## Fast Usage
```bash
chmod +x build-for-debug.sh && build-for-debug.sh
```

## Install 
try to install debian files, commands:
```bash
sudo dpkg -i kytensor-llm_2.0.0-ok15k0.7_*.deb 
sudo dpkg -i kytensor-llm-dev_2.0.0-ok15k0.7_*.deb 
```

## Use
enter into folder 'tests/kytensor', then test `llama-cli`
1. download models
```bash
cd tests/kytensor/
pip3 install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages
modelscope download --model unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M --local_dir ./model
modelscope download --model Qwen/Qwen1.5-0.5B-Chat-GGUF qwen1_5-0_5b-chat-q2_k.gguf --local_dir ./model
```

2. test `llama-cli`
```bash
chmod +x test-cli-params.sh         
./test-cli-params.sh
```