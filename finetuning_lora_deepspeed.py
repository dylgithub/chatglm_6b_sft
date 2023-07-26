# -*- coding:utf-8 -*-
# @project: ChatGLM-Finetuning
# @filename: finetuning_lora_my
"""
    文件说明：
            
"""
# from modeling_chatglm import ChatGLMForConditionalGeneration
# from tokenization_chatglm import ChatGLMTokenizer
import torch
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import argparse
import time
import deepspeed
from torch.utils.data import RandomSampler, DataLoader
from data_set import Seq2SeqDataSet, coll_fn
import os
from shutil import copy
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, \
    set_peft_model_state_dict


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        # 统计参数总量
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/spo_0.json', type=str, help='')
    parser.add_argument('--model_dir', default="/content/drive/MyDrive/model/chatglm-6b/", type=str, help='')
    # parser.add_argument('--model_dir', default="THUDM/chatglm-6b", type=str, help='')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='')
    parser.add_argument('--train_batch_size', default=1, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='output_dir_lora/', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--lora_r', type=int, default=8, help='')
    parser.add_argument('--lr', type=int, default=1e-5, help='')
    parser.add_argument('--prompt_text', type=str,
                        default="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",
                        help='')
    return parser.parse_args()


def main():
    args = set_args()
    # AutoModel可以使用自定义的模型trust_remote_code要设定为True
    # 注意这里对model的half转换，要在加载原始模型时就转换，不能在获取peft_model之后
    # 否则可能会因为精度的转换问题出现Loss为nan的情况
    # 同时需要注意half转换要在分配gpu之前
    # 打印模型参数可以发现chatglm-6B模型参数都是torch.float16类型的，这里要用.half()
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True).half()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    config = LoraConfig(r=args.lora_r,
                        lora_alpha=32,
                        target_modules=["query_key_value"],  # lora的目标位置，具体有哪些可选项可打印出源码中的key_list 注意不同的模型中的定义名称不同
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )

    model = get_peft_model(model, config)
    conf = {"train_micro_batch_size_per_gpu": args.train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-5,
                    "betas": [
                        0.9,
                        0.95
                    ],
                    "eps": 1e-8,
                    "weight_decay": 5e-4
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 1,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True
            },
            "steps_per_print": args.log_steps
            }
    print_trainable_parameters(model)
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print(name)

    train_dataset = Seq2SeqDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.prompt_text)
    print("args.train_batch_size....", args.train_batch_size)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=RandomSampler(train_dataset),
                                  collate_fn=coll_fn,
                                  drop_last=False,
                                  num_workers=0)
    model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,
                                                         model=model,
                                                         model_parameters=model.parameters())
    # optimizer and lr scheduler
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=(len(train_dataloader) * args.num_train_epochs),
    # )
    model_engine.train()
    global_step = 0
    for i_epoch in range(args.num_train_epochs):
        train_iter = iter(train_dataloader)
        for step, batch in enumerate(train_iter):
            # 数据和模型使用相同gpu
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model_engine.forward(input_ids=input_ids, labels=labels)
            loss = outputs[0]
            if conf["gradient_accumulation_steps"] > 1:
                loss = loss / conf["gradient_accumulation_steps"]
            model_engine.backward(loss)
            print(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if (step + 1) % conf["gradient_accumulation_steps"] == 0:
                model_engine.step()
                global_step += 1
            if global_step % args.log_steps == 0:
                print("loss:{}, global_step:{}".format(float(loss.item()), global_step))
    # 注意这里的模型保存方式，peft重写了model的save_pretrained方法，这里只把lora层的权重进行存储
    # 如果是用trainer进行训练，需要注意对模型保存的方法进行重写只保存lora的参数，具体参考：https://cloud.tencent.com/developer/article/2276508
    model_engine.save_pretrained(args.output_dir)
    copy(os.path.join(args.model_dir, "tokenizer_config.json"), os.path.join(args.output_dir, "tokenizer_config.json"))
    copy(os.path.join(args.model_dir, "tokenization_chatglm.py"),
         os.path.join(args.output_dir, "tokenization_chatglm.py"))
    copy(os.path.join(args.model_dir, "ice_text.model"), os.path.join(args.output_dir, "ice_text.model"))


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)
    # 模型训练时的显存占用=模型自身的参数量显存占用+batch_size*每条数据需用的显存。
    # 数据需用的显存是指反向传播时参数更新需要用到的一些数据，一般是可训练参数量的2-3倍，adam是四倍差不多, 这里只是粗略的估计
    # 占用显存大小25287MiB约25G fp16 1个epoch耗时：336.63322257995605
    # batch_size为2占用显存大小为37481MiB  差不多是12+13*2=38
