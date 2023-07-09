from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
import torch
import os
from shutil import copy


# when merging disable int8
original_model_dir = "/root/dyl/demo/ChatGLM-6B/models/chatglm-6b/"
lora_model_dir = "/root/dyl/demo/ChatGLM6b_finetuning/output_dir_lora/"
model = AutoModel.from_pretrained(original_model_dir, trust_remote_code=True).half()
tokenizer = AutoTokenizer.from_pretrained(original_model_dir, trust_remote_code=True)

## 用来检查权重是否合并成功，合并成功weight会改变
first_weight = model.base_model.layers[0].attention.query_key_value.weight
first_weight_old = first_weight.clone()
lora_model = PeftModel.from_pretrained(model, lora_model_dir)
# # 报错：A*B shape mismatch，大概率是get_peft_model错误修改了peft_config里面的fan_in_fan_out参数，某个peft的revision有这个bug
lora_model = lora_model.merge_and_unload()
lora_model.train(False)
# 验证weight值是否改变了
# # 报错：大概率peft训练有问题，检查adapter.bin大小
assert not torch.allclose(first_weight_old, first_weight), 'Weight Should Change after Lora Merge'
# # lora模型权重把原模型权重加了prefix，这里移除恢复原始key
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model.state_dict().items()
    if "lora" not in k
}
# 保存合并后的模型权重
output_dir = "output_dir_lora_merge"
os.makedirs(output_dir, exist_ok=True)
lora_model.save_pretrained(output_dir, state_dict=deloreanized_sd, max_shard_size="4GB")
copy(os.path.join(original_model_dir, "tokenizer_config.json"), os.path.join(output_dir, "tokenizer_config.json"))
copy(os.path.join(original_model_dir, "configuration_chatglm.py"), os.path.join(output_dir, "configuration_chatglm.py"))
copy(os.path.join(original_model_dir, "modeling_chatglm.py"), os.path.join(output_dir, "modeling_chatglm.py"))
copy(os.path.join(original_model_dir, "quantization.py"), os.path.join(output_dir, "quantization.py"))
copy(os.path.join(original_model_dir, "tokenization_chatglm.py"),
     os.path.join(output_dir, "tokenization_chatglm.py"))
copy(os.path.join(original_model_dir, "ice_text.model"), os.path.join(output_dir, "ice_text.model"))