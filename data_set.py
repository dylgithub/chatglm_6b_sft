# -*- coding:utf-8 -*-
# @project: ChatGPT
# @filename: data_set
# @author: 刘聪NLP
# @zhihu: https://www.zhihu.com/people/LiuCongNLP
# @contact: logcongcong@gmail.com
# @time: 2023/4/4 14:42
"""
    文件说明：
            
"""
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Seq2SeqDataSet(Dataset):
    """数据处理函数"""
    def __init__(self, data_path, tokenizer, max_len, max_src_len, prompt_text):
        # prompt_text = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
        # -3是因为需要拼接三个特殊字符[gMASK]、<sop>、<eop>
        max_tgt_len = max_len - max_src_len - 3
        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                # chatglm的token不是中文的字，是词
                # add_special_tokens = True时会在末位添加["[gMASK]", "<sop>"]
                src_tokens = tokenizer.tokenize(sample["text"])
                # print(sample["text"])
                # print(src_tokens)
                prompt_tokens = tokenizer.tokenize(prompt_text)
                # 根据限制的长度对输入进行截断
                if len(src_tokens) > max_src_len - len(prompt_tokens):
                    src_tokens = src_tokens[:max_src_len - len(prompt_tokens)]

                tgt_tokens = tokenizer.tokenize(sample["answer"])
                # 根据限制的长度对输入进行截断
                if len(tgt_tokens) > max_tgt_len:
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                # 问、答之间需要通过特殊字符进行分割，同时需要添加终止符
                # <sop>是下一个句子开始的标记
                # tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"] + tgt_tokens + ["<eop>"]
                tokens = prompt_tokens + src_tokens + [tokenizer.gmask_token, tokenizer.bos_token] + tgt_tokens + [tokenizer.eos_token]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                # prompt和问题部分不参与损失值计算
                labels = [-100] * context_length + input_ids[mask_position + 1:]
                # 根据最大长度进行后填充
                pad_len = max_len - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                # 填充部分不参与损失值计算
                labels = labels + [-100] * pad_len
                # 区分有用的部分和填充的token
                attention_mask = []
                for input_id in input_ids:
                    if input_id != tokenizer.pad_token_id:
                        attention_mask.append(True)
                    else:
                        attention_mask.append(False)
                self.all_data.append(
                    {"text": sample["text"], "answer": sample["answer"], "input_ids": input_ids, "labels": labels, "attention_mask": attention_mask})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        instance = self.all_data[item]
        return instance


def coll_fn(batch):
    input_ids_list, labels_list, attention_mask_list = [], [], []
    for instance in batch:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
        attention_mask_list.append(torch.tensor(instance["attention_mask"]))
    # dataset中已经根据设定的长度进行了长度对齐
    sample = {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=3),
            "labels": pad_sequence(labels_list, batch_first=True, padding_value=3),
            # "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=False)
            }
    # items = {k: torch.tensor(v).to("cuda:1") for k, v in sample.items()}

    return sample

if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
    from torch.utils.data import RandomSampler, DataLoader
    train_path = "data/spo_1.json"
    max_len = 100
    max_src_len = 80
    prompt_text = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
    model_dir = "/root/dyl/demo/ChatGLM-6B/models/chatglm-6b/"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    print(tokenizer.pad_token_id)
    train_dataset = Seq2SeqDataSet(train_path, tokenizer, max_len, max_src_len, prompt_text)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1,
                                  collate_fn=coll_fn,
                                  # shuffle = True,
                                  drop_last=True,
                                  num_workers=0)
    train_iter = iter(train_dataloader)
    for step, batch in enumerate(train_iter):
        if step == 1:
            print(batch["input_ids"])
            print(batch["attention_mask"])
            print(batch["labels"])
            print(batch["input_ids"].shape)
            print(batch["labels"].shape)
            print(batch["attention_mask"].shape)