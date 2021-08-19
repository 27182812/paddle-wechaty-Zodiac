
from paddlenlp.transformers import GPTChineseTokenizer

# 设置想要使用模型的名称
model_name = 'gpt-cpm-small-cn-distill'
tokenizer = GPTChineseTokenizer.from_pretrained(model_name)

import paddle




from paddlenlp.transformers import GPTForPretraining

# 一键加载中文GPT模型
model = GPTForPretraining.from_pretrained(model_name)

def chat(user_input):

    #user_input = "花间一壶酒，独酌无相亲。举杯邀明月，"

    # 将文本转为ids
    input_ids = tokenizer(user_input)['input_ids']
    #print(input_ids)

    # 将转换好的id转为tensor
    input_ids = paddle.to_tensor(input_ids, dtype='int64').unsqueeze(0)
    #print(input_ids)

    # 调用生成API升成文本
    ids, scores = model.generate(
                    input_ids=input_ids,
                    max_length=36,
                    min_length=1,
        decode_strategy='sampling',
        top_k=5,
    num_return_sequences=3)

    # print(ids)
    # print(scores)
    generated_ids = ids[0].numpy().tolist()

    # 使用tokenizer将生成的id转为文本
    generated_text = tokenizer.convert_ids_to_string(generated_ids)
    print(generated_text)
    return generated_text.rstrip(',')

if __name__ == '__main__':
    chat("你好啊,宝贝")





