# -*- coding:utf-8 -*-
# @project: ChatGLM-Finetuning
# @filename: finetuning_lora_ngpu
"""
    文件说明：单机多卡，数据并行sft
    文件的运行要通过torch.distributed.launch进行，具体参考run_ngpu.sh
    运行时需要传入参数nproc_per_node指定运行需用的进程数，一般是=你想使用的gpu个数
    参考：https://zhuanlan.zhihu.com/p/76638962

"""
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import argparse
import time
from torch.utils.data import DataLoader
from data_set import Seq2SeqDataSet, coll_fn
import os
from shutil import copy
from collections import OrderedDict
from peft import LoraConfig, get_peft_model
from torch.utils.data.distributed import DistributedSampler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def setup():
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
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
    parser.add_argument('--train_path', default='data/spo_1.json', type=str, help='')
    parser.add_argument('--model_dir', default="/root/dyl/demo/ChatGLM-6B/models/chatglm-6b/", type=str, help='')
    # parser.add_argument('--model_dir', default="THUDM/chatglm-6b", type=str, help='')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='')
    parser.add_argument('--train_batch_size', default=1, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='output_dir_lora_ngpu_temp/', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--lora_r', type=int, default=8, help='')
    parser.add_argument('--lr', type=int, default=1e-5, help='')
    # 这个参数表示当前进程对应的GPU号，系统会自动识别，注意这个参数，必须要以这种形式指定，即使代码中不使用。因为 launch 工具默认传递该参数
    parser.add_argument("--local-rank", default=os.environ['LOCAL_RANK'], type=int)
    parser.add_argument('--prompt_text', type=str,
                        default="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",
                        help='')
    return parser.parse_args()


def get_trainable_param_key(model):
    key_set = set()
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            key_set.add(str(name).replace("module.", ""))
    return key_set


def main():
    # device = "cuda:1"
    args = set_args()
    # setup()
    # world_size = dist.get_world_size()
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # if args.local_rank != -1:
    #     torch.cuda.set_device(args.local_rank)
    #     device = torch.device("cuda", args.local_rank)
    # #     print(device)
    #     dist.init_process_group(backend="nccl", init_method='env://')
    # world_size = dist.get_world_size()
    # print("world_size.........", world_size)
    # 通过Env方式进行初始化，这里Default is "env://" if no init_method`` or ``store`` is specified.

    torch.distributed.init_process_group(backend='nccl')
    # 获得当前进程使用的gpu号
    local_rank = torch.distributed.get_rank()
    # print("local_rank......", local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # print("device.....", device)
    # AutoModel可以使用自定义的模型trust_remote_code要设定为True
    # 注意这里对model的half转换，要在加载原始模型时就转换，不能在获取peft_model之后
    # 否则可能会因为精度的转换问题出现Loss为nan的情况
    # 同时需要注意half转换要在分配gpu之前
    # 打印模型参数可以发现chatglm-6B模型参数都是torch.float16类型的，这里要用.half()
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True).half().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    config = LoraConfig(r=args.lora_r,
                        lora_alpha=32,
                        target_modules=["query_key_value"],  # lora的目标位置，具体有哪些可选项可打印出源码中的key_list
                        lora_dropout=0.1,
                        bias="none",
                        task_type="CAUSAL_LM",
                        inference_mode=False,
                        )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                output_device=local_rank)
    # print("DistributedDataParallel_model type........", type(model))
    train_dataset = Seq2SeqDataSet(args.train_path, tokenizer, args.max_len, args.max_src_len, args.prompt_text)

    train_sampler = DistributedSampler(train_dataset)
    # 通过分布式采样自动会进行随机
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler,
                                  collate_fn=coll_fn)

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_train_epochs),
    )

    for i_epoch in range(args.num_train_epochs):
        train_sampler.set_epoch(i_epoch)
        model.train()
        train_iter = iter(train_dataloader)
        for step, batch in enumerate(train_iter):
            # 数据和模型使用相同gpu
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            print(loss)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        # 等待所有进程计算完毕
        # torch.cuda.synchronize(device)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # 多个gpu上的模型只需要保存一个便可
    if local_rank == 0:
        # 只保存lora部分的可训练参数
        # 注意这里的模型保存方式，peft重写了model的save_pretrained方法，这里只把lora层的权重进行存储
        model.module.save_pretrained(args.output_dir)
        copy(os.path.join(args.model_dir, "tokenizer_config.json"),
             os.path.join(args.output_dir, "tokenizer_config.json"))
        copy(os.path.join(args.model_dir, "tokenization_chatglm.py"),
             os.path.join(args.output_dir, "tokenization_chatglm.py"))
        copy(os.path.join(args.model_dir, "ice_text.model"), os.path.join(args.output_dir, "ice_text.model"))
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(end_time - start_time)
    # 模型训练时的显存占用=模型自身的参数量显存占用+batch_size*每条数据需用的显存。
    # 数据需用的显存是指反向传播时参数更新需要用到的一些数据，一般是可训练参数量的2-3倍，adam是四倍差不多, 这里只是粗略的估计
    # 占用显存大小25547MiB约25G fp16 1个epoch耗时：198.83175444602966
    # batch_size为2占用显存大小为37481MiB  差不多是12+13*2=38
