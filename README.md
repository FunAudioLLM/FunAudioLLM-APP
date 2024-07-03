# funaudiollm-app repo
Welcome to the funaudiollm-app repository! This project hosts two exciting applications leveraging advanced audio understand and speech generation models to bring your audio experiences to life:

**Voice Chat** :  This application is designed to provide an interactive and natural chatting experience, making it easier to adopt sophisticated AI-driven dialogues in various settings.

**Voice Translation**: Break down language barriers with our real-time voice translation tool. This application seamlessly translates spoken language on the fly, allowing for effective and fluid communication between speakers of different languages.


## Install

**Clone and install**

- Clone the repo
``` sh
git clone --recursive URL
# If you failed to clone submodule due to network failures, please run following command until success
cd funaudiollm-app
git submodule update --init --recursive
```

- prepare environments according to cosyvoice & sensevoice repo. then, execute the code below.
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


