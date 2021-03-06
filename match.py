import numpy as np
import os
import time
import paddle
import paddle.nn.functional as F
from paddlenlp.datasets import load_dataset
import paddlenlp
# 为了后续方便使用，我们给 convert_example 赋予一些默认参数
from functools import partial
from paddlenlp.data import Stack, Pad, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

import paddle.nn as nn

# 我们基于 ERNIE1.0 模型结构搭建 Point-wise 语义匹配网络
# 所以此处先定义 ERNIE1.0 的 pretrained_model
pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained('ernie-1.0')


class PointwiseMatching(nn.Layer):

    # 此处的 pretained_model 在本例中会被 ERNIE1.0 预训练模型初始化
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # 语义匹配任务: 相似、不相似 2 分类任务
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):
        # 此处的 Input_ids 由两条文本的 token ids 拼接而成
        # token_type_ids 表示两段文本的类型编码
        # 返回的 cls_embedding 就表示这两段文本经过模型的计算之后而得到的语义表示向量
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding)

        # 基于文本对的语义表示向量进行 2 分类任务
        logits = self.classifier(cls_embedding)
        probs = F.softmax(logits)

        return probs

def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    # 在预测或者评估阶段，不返回 label 字段
    else:
        return input_ids, token_type_ids

def read_text_pair(data_path):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:

            data = line.rstrip().split(" ")
            # print(data)
            # print(len(data))
            if len(data) != 2:
                continue
            yield {'query': data[0], 'title': data[1]}

def predict(model, data_loader):
    batch_probs = []

    # 预测阶段打开 eval 模式，模型中的 dropout 等操作会关掉
    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
            batch_prob = model(
                input_ids=input_ids, token_type_ids=token_type_ids).numpy()
            # print("111",batch_prob)
            batch_probs.append(batch_prob)
        batch_probs = np.concatenate(batch_probs, axis=0)

        return batch_probs
# 预测数据的转换函数
    # predict 数据没有 label, 因此 convert_exmaple 的 is_test 参数设为 True
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=512,
    is_test=True)

# 预测数据的组 batch 操作
# predict 数据只返回 input_ids 和 token_type_ids，因此只需要 2 个 Pad 对象作为 batchify_fn
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
): [data for data in fn(samples)]

pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained("ernie-1.0")

model = PointwiseMatching(pretrained_model)

# 刚才下载的模型解压之后存储路径为 ./pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams
state_dict = paddle.load("pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams")
model.set_dict(state_dict)

def start():
    # 加载预测数据
    predict_ds = load_dataset(
        read_text_pair, data_path="./predict.txt", lazy=False)
    # for i in predict_ds:
    #     print(i)
    batch_sampler = paddle.io.BatchSampler(predict_ds, batch_size=32, shuffle=False)

    # 生成预测数据 data_loader
    predict_data_loader = paddle.io.DataLoader(
        dataset=predict_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

    # 执行预测函数
    y_probs = predict(model, predict_data_loader)

    # 根据预测概率获取预测 label
    y_preds = np.argmax(y_probs, axis=1)
    print(y_preds)
    return y_preds[-1]

    # predict_ds = load_dataset(
    #     read_text_pair, data_path="./predict.txt", lazy=False)
    #
    # for idx, y_pred in enumerate(y_preds):
    #     text_pair = predict_ds[idx]
    #     text_pair["pred_label"] = y_pred
    #     print(text_pair)


if __name__ == '__main__':
    start()


