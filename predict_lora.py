# -*- coding:utf-8 -*-
# @project: ChatGLM-Finetuning
# @filename: predict_lora
"""
    文件说明：
            
"""
import torch
import json
from peft import PeftModel
from tqdm import tqdm
import time
import os
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/spo_1.json', type=str, help='')
    parser.add_argument('--device', default='1', type=str, help='')
    parser.add_argument('--ori_model_dir',
                        default="/root/dyl/demo/ChatGLM-6B/models/chatglm-6b/", type=str,
                        help='')
    parser.add_argument('--model_dir',
                        default="/root/dyl/demo/ChatGLM-Finetuning/output_dir_lora/", type=str,
                        help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--prompt_text', type=str,
                        default="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",
                        help='')
    return parser.parse_args()


def main():
    args = set_args()
    # 先加载原始模型
    model = AutoModel.from_pretrained(args.ori_model_dir, trust_remote_code=True).half().to(
        "cuda:{}".format(args.device))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model.eval()
    # 再加载lora层的参数
    model = PeftModel.from_pretrained(model, args.model_dir)
    model.half().to("cuda:{}".format(args.device))
    model.eval()
    # 注意以上加载方式会增加推理的时间，因为增加了lora层的参数
    # 另一种加载方法另一个没有推理延时的方案，是先把lora权重和原始模型权重进行合并，把合并后的参数存储成新的bin文件，然后和加载常规模型一样加载合并后的模型参数进行推理
    # 具体可参考https://cloud.tencent.com/developer/article/2276508  模型推理部分
    save_data = []
    f1 = 0.0
    max_tgt_len = args.max_len - args.max_src_len - 3
    s_time = time.time()
    with open(args.test_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(tqdm(fh, desc="iter")):
            with torch.no_grad():
                sample = json.loads(line.strip())
                src_tokens = tokenizer.tokenize(sample["text"])
                prompt_tokens = tokenizer.tokenize(args.prompt_text)

                if len(src_tokens) > args.max_src_len - len(prompt_tokens):
                    src_tokens = src_tokens[:args.max_src_len - len(prompt_tokens)]

                tokens = prompt_tokens + src_tokens + ["[gMASK]", "<sop>"]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                # input_ids = tokenizer.encode("帮我写个快排算法")

                input_ids = torch.tensor([input_ids]).to("cuda:{}".format(args.device))
                generation_kwargs = {
                    "min_length": 5,
                    "max_new_tokens": max_tgt_len,
                    "top_p": 0.7,
                    "temperature": 0.95,
                    "do_sample": False,
                    "num_return_sequences": 1,
                }
                response = model.generate(input_ids=input_ids,
                                          min_length=5,
                                          max_new_tokens=max_tgt_len,
                                          top_p=0.7,
                                          temperature=0,
                                          do_sample=False,
                                          num_return_sequences=1)
                # print(response)
                res = []
                for i_r in range(generation_kwargs["num_return_sequences"]):
                    outputs = response.tolist()[i_r][input_ids.shape[1]:]
                    r = tokenizer.decode(outputs).replace("<eop>", "")
                    print(r)
                    res.append(r)
                pre_res = [rr for rr in res[0].split("\n") if len(rr.split("_")) == 3]
                real_res = sample["answer"].split("\n")
                same_res = set(pre_res) & set(real_res)
                if len(set(pre_res)) == 0:
                    p = 0.0
                else:
                    p = len(same_res) / len(set(pre_res))
                r = len(same_res) / len(set(real_res))
                if (p + r) != 0.0:
                    f = 2 * p * r / (p + r)
                else:
                    f = 0.0
                f1 += f
                save_data.append(
                    {"text": sample["text"], "ori_answer": sample["answer"], "gen_answer": res[0], "f1": f})

    e_time = time.time()
    print("总耗时：{}s".format(e_time - s_time))
    print(f1 / 50)
    save_path = os.path.join(args.model_dir, "ft_pt_answer.json")
    fin = open(save_path, "w", encoding="utf-8")
    json.dump(save_data, fin, ensure_ascii=False, indent=4)
    fin.close()


if __name__ == '__main__':
    main()
