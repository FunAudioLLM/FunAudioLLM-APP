# funaudiollm-app repo
Welcome to the funaudiollm-app repository! This project hosts two exciting applications leveraging advanced audio understand and speech generation models to bring your audio experiences to life:

**Voice Chat** :  This application is designed to provide an interactive and natural chatting experience, making it easier to adopt sophisticated AI-driven dialogues in various settings.

**Voice Translation**: Break down language barriers with our real-time voice translation tool. This application seamlessly translates spoken language on the fly, allowing for effective and fluid communication between speakers of different languages.

For Details, visit [FunAudioLLM Homepage](https://fun-audio-llm.github.io/), [CosyVoice Paper](https://fun-audio-llm.github.io/pdf/CosyVoice_v1.pdf), [FunAudioLLM Technical Report](https://fun-audio-llm.github.io/pdf/FunAudioLLM.pdf)

For `CosyVoice`, visit [CosyVoice repo](https://github.com/FunAudioLLM/CosyVoice) and [CosyVoice space](https://www.modelscope.cn/studios/iic/CosyVoice-300M).

For `SenseVoice`, visit [SenseVoice repo](https://github.com/FunAudioLLM/SenseVoice) and [SenseVoice space](https://www.modelscope.cn/studios/iic/SenseVoice).

## Install

**Clone and install**

- Clone the repo and submodules
``` sh
git clone --recursive URL
# If you failed to clone submodule due to network failures, please run following command until success
cd funaudiollm-app
git submodule update --init --recursive
```

- prepare environments in the submodules according to [cosyvoice](https://github.com/FunAudioLLM/CosyVoice) & [sensevoice](https://github.com/FunAudioLLM/SenseVoice) repo. If you have already prepared the aforementioned resources elsewhere, you can also try modifying the code related to resource path configuration in the app.py file (line 15-18).


- execute the code below.
``` sh
pip install -r requirements.txt
```

## Basic Usage
**prepare**


[dashscope](https://dashscope.aliyun.com/) api token.

[pem file](https://blog.csdn.net/liuchenbaidu/article/details/136722001)


**voice chat**

``` sh
cd voice_chat
sudo CUDA_VISIBLE_DEVICES="0" DS_API_TOKEN="YOUR-DS-API-TOKEN" python app.py >> ./log.txt
```
https://YOUR-IP-ADDRESS:60001/

**voice translation**

``` sh
cd voice_translation
sudo CUDA_VISIBLE_DEVICES="0" DS_API_TOKEN="YOUR-DS-API-TOKEN" python app.py >> ./log.txt
```
https://YOUR-IP-ADDRESS:60002/


