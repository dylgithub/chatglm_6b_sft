from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import torch

# when merging disable int8
original_model_dir = "/root/dyl/demo/ChatGLM-6B/models/chatglm-6b/"
lora_model_dir = "/root/dyl/demo/ChatGLM6b_finetuning/output_dir_lora_merge"
model = AutoModel.from_pretrained(lora_model_dir, trust_remote_code=True).half()
tokenizer = AutoTokenizer.from_pretrained(original_model_dir, trust_remote_code=True)

## 用来检查权重是否合并成功，合并成功weight会改变
# first_weight = model.base_model.layers[0].attention.query_key_value.weight
# first_weight_old = first_weight.clone()
device = torch.device("cuda:1")
# model = PeftModel.from_pretrained(model, lora_model_dir)
model = model.to(device)


# # 返回的不是新的模型，而是在原始模型上加了adapter层
# lora_model = PeftModel.from_pretrained(
#     model,
#     "./lora_ckpt",
#     device_map={"": "cpu"},
#     torch_dtype=torch.float16,
# )
# # 报错：A*B shape mismatch，大概率是get_peft_model错误修改了peft_config里面的fan_in_fan_out参数，某个peft的revision有这个bug
# lora_model = lora_model.merge_and_unload()
# lora_model.train(False)
#
# # 报错：大概率peft训练有问题，检查adapter.bin大小
# assert not torch.allclose(first_weight_old, first_weight), 'Weight Should Change after Lora Merge'
#
# # lora模型权重把原模型权重加了prefix，这里移除恢复原始key
# deloreanized_sd = {
#     k.replace("base_model.model.", ""): v
#     for k, v in lora_model.state_dict().items()
#     if "lora" not in k
# }
# # 保存合并后的模型权重
# lora_model.save_pretrained(output_dir, state_dict=deloreanized_sd)
#print(model)

# 未合并占用的显存是13031
#model = model.to(device)

text = "Hello, my name is "
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=30, top_p=0.85)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("\n------------------------------------------------\nInput: ")

line = input()
while line:
  inputs = tokenizer(line, return_tensors="pt").to(device)
  outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=30, top_p=0.85)
  print("Output: ",tokenizer.decode(outputs[0], skip_special_tokens=True))
  print("\n------------------------------------------------\nInput: ")
  line = input()

