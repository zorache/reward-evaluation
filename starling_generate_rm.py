import os
os.environ['HF_HOME'] = '/cmlscratch/zche/.cache/huggingface'
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from starling_rm import GPTRewardModel
import numpy as np
import json
import math


reward_device = "cuda"
reward_batch_size = 3


reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf")
reward_tokenizer = reward_model.tokenizer
reward_tokenizer.truncation_side = "left"
# print("passed 1")

directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha")
for fpath in os.listdir(directory):
    if fpath.endswith(".pt") or fpath.endswith("model.bin"):
        checkpoint = os.path.join(directory, fpath)
        break
   
reward_model.load_state_dict(torch.load(checkpoint), strict=False)
reward_model.eval().requires_grad_(False)

reward_model = reward_model.to(reward_device)

def get_reward(samples):
    """samples: List[str]"""
    input_ids = []
    attention_masks = []
    encodings_dict = reward_tokenizer(
        # samples,
        samples,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encodings_dict["input_ids"].to(reward_device)
    attention_masks = encodings_dict["attention_mask"]
    mbs = reward_batch_size
    out = []
    for i in range(math.ceil(len(samples) / mbs)):
        rewards = reward_model(input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs])
        out.extend(rewards)
    return torch.hstack(out)


# test_sample = ["<s>[INST] Hello? </s> [/INST] Hi, how can I help you?</s>"] 
# reward_for_test_sample = get_reward(test_sample)
# print(reward_for_test_sample)


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYS_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
WRAPPED_PROMPT = B_INST + " " + B_SYS + SYS_PROMPT + E_SYS + "%s" + " " + E_INST


# This is the format for page 1 of the google sheets, no system message
def generate_format_standard(instruction, response, system_message = None, tokenizer = reward_tokenizer):
    template = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': instruction},{'role': 'assistant', 'content': response}] if system_message else [{'role': 'user', 'content': instruction}, {'role': 'assistant', 'content': response}]
    chat = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
    return chat


# This is the format for page 2 of the google sheets
def generate_format(instruction, response):
    chat = "<s>"+ B_INST +" "+instruction+"</s>" +" "+ E_INST +" "+ response +"</s>"
    return [chat]

with open("harmbench-GCG-starling-checked-jailbreak.json","r") as f:
    data = json.load(f)

x_y_safe_lst =  []
x_adv_y_safe_lst =  []
x_adv_y_unsafe_lst = []
x_y_unsafe_lst = []


def make_shorter(response):
    response = " ".join(response.split(" ")[:3])
    return response

make_short = True 

if make_short:
    for i in range(len(data)):
        item = data[i]
        target = item["target"]
        prompt = item["prompt"]
        adv_responses = item["response"]
        baseline = item["baseline"]
        x_y_safe = get_reward(generate_format(target,make_shorter(baseline))).item()
        x_y_safe_lst.append(x_y_safe)
        x_adv_y_safe = get_reward(generate_format(prompt,make_shorter(baseline))).item()
        x_adv_y_safe_lst.append(x_adv_y_safe)
        x_adv_y_unsafe_0 = get_reward(generate_format(prompt, make_shorter(adv_responses[0]))).item()
        x_adv_y_unsafe_1 = get_reward(generate_format(prompt, make_shorter(adv_responses[1]))).item()
        x_adv_y_unsafe_2 = get_reward(generate_format(prompt, make_shorter(adv_responses[2]))).item()
        x_adv_y_unsafe_lst.append((x_adv_y_unsafe_0+x_adv_y_unsafe_1+x_adv_y_unsafe_2)/3)
        x_y_unsafe_0 = get_reward(generate_format(target, make_shorter(adv_responses[0]))).item()
        x_y_unsafe_1 = get_reward(generate_format(target, make_shorter(adv_responses[1]))).item()
        x_y_unsafe_2 = get_reward(generate_format(target, make_shorter(adv_responses[2]))).item()
        x_y_unsafe_lst.append((x_y_unsafe_0+ x_y_unsafe_1+x_y_unsafe_2)/3)
        dct = {"x_y_safe":x_y_safe,"x_adv_y_safe":x_adv_y_safe,"x_adv_y_unsafe_0":x_adv_y_unsafe_0,"x_adv_y_unsafe_1":x_adv_y_unsafe_1,"x_adv_y_unsafe_2":x_adv_y_unsafe_2,"x_y_unsafe_0":x_y_unsafe_0,"x_y_unsafe_1":x_y_unsafe_1,"x_y_unsafe_2":x_y_unsafe_2}
        data[i]["rewards"]=dct
else:
    for i in range(len(data)):
        item = data[i]
        target = item["target"]
        prompt = item["prompt"]
        adv_responses = item["response"]
        baseline = item["baseline"]
        x_y_safe = get_reward(generate_format(target,baseline)).item()
        x_y_safe_lst.append(x_y_safe)
        x_adv_y_safe = get_reward(generate_format(prompt,baseline)).item()
        x_adv_y_safe_lst.append(x_adv_y_safe)
        x_adv_y_unsafe_0 = get_reward(generate_format(prompt,adv_responses[0])).item()
        x_adv_y_unsafe_1 = get_reward(generate_format(prompt,adv_responses[1])).item()
        x_adv_y_unsafe_2 = get_reward(generate_format(prompt,adv_responses[2])).item()
        x_adv_y_unsafe_lst.append((x_adv_y_unsafe_0+x_adv_y_unsafe_1+x_adv_y_unsafe_2)/3)
        x_y_unsafe_0 = get_reward(generate_format(target,adv_responses[0])).item()
        x_y_unsafe_1 = get_reward(generate_format(target,adv_responses[1])).item()
        x_y_unsafe_2 = get_reward(generate_format(target,adv_responses[2])).item()
        x_y_unsafe_lst.append((x_y_unsafe_0+ x_y_unsafe_1+x_y_unsafe_2)/3)
        dct = {"x_y_safe":x_y_safe,"x_adv_y_safe":x_adv_y_safe,"x_adv_y_unsafe_0":x_adv_y_unsafe_0,"x_adv_y_unsafe_1":x_adv_y_unsafe_1,"x_adv_y_unsafe_2":x_adv_y_unsafe_2,"x_y_unsafe_0":x_y_unsafe_0,"x_y_unsafe_1":x_y_unsafe_1,"x_y_unsafe_2":x_y_unsafe_2}
        data[i]["rewards"]=dct




x_y_safe_lst = np.array(x_y_safe_lst)
x_adv_y_safe_lst = np.array(x_adv_y_safe_lst)
x_adv_y_unsafe_lst = np.array(x_adv_y_unsafe_lst)
x_y_unsafe_lst = np.array(x_y_unsafe_lst)

# np.savetxt('x_y_safe_lst.out', x_y_safe_lst, delimiter=',') 
# np.savetxt('x_adv_y_safe_lst.out', x_adv_y_safe_lst, delimiter=',') 
# np.savetxt('x_adv_y_unsafe_lst.out', x_adv_y_unsafe_lst, delimiter=',') 
# np.savetxt('x_y_unsafe_lst.out', x_y_unsafe_lst, delimiter=',') 


# with open("harmbench-GCG-starling-checked-jailbreak-2.json", "w") as file:
#     json.dump(data, file, indent=4)

np.savetxt('starling-outputs-2-short.csv', np.column_stack((x_y_safe_lst, x_adv_y_safe_lst,x_adv_y_unsafe_lst,x_y_unsafe_lst)), delimiter=",", fmt='%s', header="x_y_safe, x_adv_y_safe, x_adv_y_unsafe, x_y_unsafe")

