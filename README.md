# paddle-wechaty-Zodiac
AI创造营 ：Metaverse启动机之重构现世，结合PaddlePaddle 和 Wechaty 创造自己的聊天机器人

# 12星座若穿越科幻剧，会拥有什么超能力呢？快来迎接你的专属超能力吧！

* 现在很多年轻人都喜欢看科幻剧，像是复仇者系列，里面有很多英雄、超能力者，这些人都是我们的青春与情怀，那么12星座若穿越科幻剧，会分别拥有什么超能力呢？
* 利用paddle提供的文本匹配和对话闲聊模型结合wechaty进行构建
* 除了获得专属超能力外，还可以查看今日运势，外加寂寞无聊时找个“算命大师”聊聊天
![](https://ai-studio-static-online.cdn.bcebos.com/dbe0acb38fbe4a4091fa5efee461e4f13035021608ce414e8d86e0b26b742f91)



## 效果展示
![](https://ai-studio-static-online.cdn.bcebos.com/09bda8a9e6a94415a7bf6a18ee5dff19a677bde4ce5a4ba4976f59e84634e683)
![](https://ai-studio-static-online.cdn.bcebos.com/7812ed9913aa4a3883f86b02f9dcaee798f5b977ea464815a75fc4d7041da7b5)
![](https://ai-studio-static-online.cdn.bcebos.com/77819a0a40fc422daf99b3df96650892c09f335350344fe1827b679e2e135185)

## 本项目的实现过程
### 云服务器部分
- 参考[https://aistudio.baidu.com/aistudio/projectdetail/2279551](http://https://aistudio.baidu.com/aistudio/projectdetail/2279551)

- 我用的阿里云的云服务器，也可以考虑其他云服务或者是外网可访问的服务器资源。

- 进入服务器终端，在终端输入以下命令（注：确保输入的端口是对外开放的，WECHATY_TOKEN请填写自己的token）


```python
$ apt update

$ apt install docker.io

$ docker pull wechaty/wechaty:latest

$ export WECHATY_LOG="verbose"

$ export WECHATY_PUPPET="wechaty-puppet-wechat"

$ export WECHATY_PUPPET_SERVER_PORT="8080"

$ export WECHATY_TOKEN="puppet_padlocal_xxxxxx" # 这里输入你自己的token

$ docker run -ti --name wechaty_puppet_service_token_gateway --rm -e WECHATY_LOG -e WECHATY_PUPPET -e WECHATY_TOKEN -e WECHATY_PUPPET_SERVER_PORT -p "$WECHATY_PUPPET_SERVER_PORT:$WECHATY_PUPPET_SERVER_PORT" wechaty/wechaty:latest
```

- 输入网址: https://api.chatie.io/v0/hosties/xxxxxx (后面的xxxxxx是自己的token)，如果返回了服务器的ip地址以及端口号，就说明运行成功了   

- 运行后会输出一大堆东西，找到一个Online QR Code: 的地址点击进去，会出现二维码，微信扫码登录，最终手机上显示桌面微信已登录，即可。


### 环境安装

```python
!pip install -U paddlepaddle -i https://mirror.baidu.com/pypi/simple
!python -m pip install --upgrade paddlenlp -i https://pypi.org/simple
!pip install --upgrade pip
!pip install --upgrade sentencepiece 
!pip install wechaty
```

### 文本匹配部分

- 文本语义匹配是NLP最基础的任务之一，简单来说就是判断两段文本的语义相似度。应用场景广泛，比如搜索引擎、智能问答、知识检索、信息流推荐等。

- 为什么要用上这个功能呢，因为如果我们直接基于关键词匹配去判断用户需求的话，可能会出现理解错误的情况。比如如果用户输入“我可讨厌星座了”，但是聊天机器人可能还是会给用户展示星座超能力；如果直接限于关键词“星座”严格匹配的话，那用户如果不小心多输入一个字或者标点符号都不能实现想要的功能，太不友好了。因此本项目利用文本匹配技术来判断用户是否真实需要查看星座未来超能力的功能。

- 本次项目基于 PaddleNLP，使用百度开源的预训练模型 ERNIE1.0，构建语义匹配模型，来判断 2 个文本语义是否相同。

- 从头训练一个模型的关键步骤有数据加载、数据预处理、模型搭建、模型训练和评估，具体可参照[https://aistudio.baidu.com/aistudio/projectdetail/1972174](https://aistudio.baidu.com/aistudio/projectdetail/1972174)，
在这我们就直接调用已经训练好的语义匹配模型进行应用。


- 下载已经训练好的语义匹配模型, 并解压


```python
! wget https://paddlenlp.bj.bcebos.com/models/text_matching/pointwise_matching_model.tar
! tar -xvf pointwise_matching_model.tar
```

- 具体代码部分（match.py文件）


### 对话闲聊部分
- 近年来，人机对话系统受到了学术界和产业界的广泛关注。开放域对话系统希望机器可以流畅自然地与人进行交互，既可以进行日常问候类的闲聊，又可以完成特定功能。
- 随着深度学习技术的不断发展，聊天机器人变得越来越智能。我们可以通过机器人来完成一些机械性的问答工作，也可以在闲暇时和智能机器人进行对话，他们的出现让生活变得更丰富多彩。
- 本项目载入该功能，也是希望人们可以在寂寞无聊的时候有个聊天的小伙伴，虽然有时候他可能会不知所云，但他永远会在那等你。

- 具体代码部分（chat.py文件）

- PaddleNLP针对生成式任务提供了generate()函数，内嵌于PaddleNLP所有的生成式模型。支持Greedy Search、Beam Search和Sampling解码策略，用户只需指定解码策略以及相应的参数即可完成预测解码，得到生成的sequence的token ids以及概率得分。
- PaddleNLP对于各种预训练模型已经内置了相应的tokenizer，指定想要使用的模型名字即可加载对应的tokenizer。
- PaddleNLP提供了GPT,UnifiedTransformer等中文预训练模型，可以通过预训练模型名称完成一键加载。这次用的是一个小的中文GPT预训练模型。其他预训练模型请参考[模型列表](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html)。

### 主函数部分（main.py）
- 星座今日运势需要自己申请一下接口，网址：[星座运势](https://www.tianapi.com/apiview/78)，将你申请到的APIKEY填入values['key']
- 运行本函数的时候，不要忘记云服务器也要开启哦，这样你的微信号才能变身为算命大师哦，不然只能在本地感受了。

## 后记

**项目可改进的部分**

- 闲聊模型不够强大，会出现不知所云的情况，又或者更偏向于文本续写而不是对话。甚至会出现不太好的话语，这应该是训练语料导致的，模型预训练时语料没有清洗，以后可以考虑用干净的更符合本项目的对话语料微调。
- 对话语义匹配的调用方式过于单调，希望之后可以改进。
- 功能还不够多，结合paddle提供的优秀资源还可以创造出更多更好玩的功能。

**互相交流进步**

- 本人研究大方向为NLP，欢迎小伙伴一起交流进步。这是我我收集的NLP资料库： [https://github.com/27182812/NLP-dataset](https://github.com/27182812/NLP-dataset)

- 觉得不错的话给我一个Star哦*_*

**参考资料**

- AI Studio官网: [https://aistudio.baidu.com/aistudio/index](https://aistudio.baidu.com/aistudio/index)

- 『NLP经典项目集』用PaddleNLP能做什么：[https://aistudio.baidu.com/aistudio/projectdetail/1535371](https://aistudio.baidu.com/aistudio/projectdetail/1535371)

- 『NLP打卡营』实践课11 动手搭建中文闲聊机器人：[https://aistudio.baidu.com/aistudio/projectdetail/2017173](https://aistudio.baidu.com/aistudio/projectdetail/2017173)

- python-wechaty: [https://github.com/wechaty/python-wechaty](https://github.com/wechaty/python-wechaty)

- python-wechaty-getting-started: [https://github.com/wechaty/python-wechaty-getting-started](https://github.com/wechaty/python-wechaty-getting-started)

- 教你用AI Studio+wechaty+阿里云白嫖一个智能微信机器人:[https://aistudio.baidu.com/aistudio/projectdetail/2279551](https://aistudio.baidu.com/aistudio/projectdetail/2279551)


**最后的最后，觉得项目写的好的话，记得fork和爱心哦，谢谢啦！！！**
