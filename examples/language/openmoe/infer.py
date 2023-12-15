from argparse import ArgumentParser

import torch
from model.modeling_openmoe import OpenMoeForCausalLM, set_openmoe_args
from model.openmoe_policy import OpenMoeForCausalLMPolicy
from transformers import T5Tokenizer
from transformers.models.llama import LlamaConfig

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.moe.manager import MOE_MANAGER
from colossalai.testing import rerun_if_address_is_in_use, spawn


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="base", type=str, help="model path", choices=["base", "8b", "test"])
    parser.add_argument("--ckpt_path", default=None, type=str, help="ckpt path")
    parser.add_argument("--num_gpus", default=1, type=int, help="number of GPUs for inference")
    return parser.parse_args()


##################################
#           1 GPU inference     #
##################################


def inference(args):
    tokenizer = T5Tokenizer.from_pretrained("google/umt5-small")
    if args.model == "test":
        config = LlamaConfig.from_pretrained("hpcaitech/openmoe-base")
        set_openmoe_args(config,
                         num_experts=config.num_experts,
                         moe_layer_interval=config.moe_layer_interval,
                         enable_kernel=True)
        model = OpenMoeForCausalLM(config)
    else:
        config = LlamaConfig.from_pretrained(f"hpcaitech/openmoe-{args.model}")
        set_openmoe_args(config,
                         num_experts=config.num_experts,
                         moe_layer_interval=config.moe_layer_interval,
                         enable_kernel=False)
        model = OpenMoeForCausalLM(config)
        if args.ckpt_path is not None:
            ckpt_path = args.ckpt_path
        assert ckpt_path is not None, "Please specify the checkpoint path"
        ckpt = torch.load(ckpt_path)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("module."):
                state_dict[key[7:]] = value
            else:
                state_dict[key] = value
        model.load_state_dict(state_dict)
        # model = OpenMoeForCausalLM.from_pretrained(f"hpcaitech/openmoe-{args.model}", config=config)
    model = model.eval().bfloat16()
    model = model.to(torch.cuda.current_device())

    # input examples for yizhongw datasets
    # inputs = [
    #     """In this task, you are given two sets, and a question. You need to find whether an element is at the intersection of two given sets. A Set is shown by two curly braces and comma-separated numbers inside, like {1, 2, 3}. The intersection of two given sets is the largest set which contains all the elements that are common to both sets. An element is at the intersection of two given sets, A and B, if common to both A and B. Classify your answers into 'Yes' or 'No'. Input: Set1: '{2, 6, 8, 9, 11, 13, 18}', Set2: '{2, 12}'. Is the element '9' in the intersection of Set1 and Set2 ? Output:""",
    #     """In this task you are given a question. You need to generate an answer to the question. Input: Question:What is the opposite side from starboard on a ship? Output:""",
    #     """In this task, you are given a list of integers and an integer k. You need to add integer k to each element in the list and return the updated list. Input: [133, 281, 294, 97, 66, 269, 39, 92, 25, 162, 16, 226, 76, 119, 268, 12, 296, 103, 211, 163] k=1. Output:""",
    #     """In this task, you are given a country name and you need to return the capital city of the given country. Input: Falkland Islands. Output:"""
    # ]
    
    # input examples for wikitext-2-v1
    inputs = [
        """Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . <unk> the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " <unk> Raven " .""",
        """Troops are divided into five classes : Scouts , <unk> , Engineers , <unk> and Armored Soldier . <unk> can switch classes by changing their assigned weapon . Changing class does not greatly affect the stats gained while in a previous class . With victory in battle , experience points are awarded to the squad , which are distributed into five different attributes shared by the entire squad , a feature differing from early games ' method of distributing to different unit types .""",
        """The plain maskray generally hunts at the surface of the bottom substrate , rather than digging for prey . Its diet consists predominantly of <unk> shrimp and polychaete worms . Small bony fishes are also eaten , along with the occasional <unk> <unk> or <unk> . Larger rays consume a greater variety of prey and relatively more polychaete worms when compared to smaller rays . This species is <unk> by the <unk> <unk> <unk> .""",
    ]
    input_str = inputs[-1][:100]
    print(f"Input: \n{input_str}\n")
    # print("model config: ", model.config)

    input_ids = tokenizer("<pad>" + input_str, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.input_ids.to(torch.cuda.current_device())
    generation_output = model.generate(input_ids, use_cache=True, do_sample=True, max_new_tokens=64)
    out = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    print(f"output: \n{out}\n")


##################################
#      Multi-GPU inference       #
##################################

def get_model(args, parallel):
    config = LlamaConfig.from_pretrained(f"hpcaitech/openmoe-{args.model}")
    set_openmoe_args(
        config,
        num_experts=config.num_experts,
        moe_layer_interval=config.moe_layer_interval,
        enable_kernel=False
    )
    model = OpenMoeForCausalLM(config)
    optim = torch.optim.Adam(model.parameters())

    if parallel == None:
        plugin = MoeHybridParallelPlugin(
            precision="bf16",
            tp_size=1,
            pp_size=1,
            zero_stage=2,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "ep":
        plugin = MoeHybridParallelPlugin(
            precision="bf16",
            tp_size=1,
            pp_size=1,
            zero_stage=2,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "ep_zero":
        plugin = MoeHybridParallelPlugin(
            precision="bf16",
            tp_size=1,
            pp_size=1,
            zero_stage=2,
            extra_dp_size=2,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    elif parallel == "hybrid":
        plugin = MoeHybridParallelPlugin(
            precision="bf16",
            tp_size=1,
            pp_size=2,
            zero_stage=1,
            microbatch_size=1,
            custom_policy=OpenMoeForCausalLMPolicy(),
        )
    booster = Booster(plugin=plugin)
    model, optim, _, _, _ = booster.boost(model=model, optimizer=optim)
    return model, booster, optim

def inference_single(args, rank, parallel):
    if parallel == None:
        MOE_MANAGER.setup(
            parallel=None,
        )
    elif parallel == "ep":
        MOE_MANAGER.setup(
            parallel="EP",
        )
    elif parallel == "ep_zero":
        MOE_MANAGER.setup(
            parallel="EP",
            max_ep_size=2,
        )
    elif parallel == "hybrid":
        MOE_MANAGER.setup(
            parallel="EP",
            mode="fixed",
            fixed_dp_size=1,
            fixed_ep_size=2,
            fixed_pp_size=2,
        )
    if args.ckpt_path is None:
        ckpt_path = "/home/nus-hx/code/ColossalAI/examples/language/openmoe/outputs/2023.12.11-13.11.06"
    else:
        ckpt_path = args.ckpt_path
    model, booster, optim = get_model(args, parallel)
    booster.load_model(model, ckpt_path)
    booster.save_model(model, "/home/nus-hx/code/ColossalAI/examples/language/openmoe/outputs/2023.12.14-04.33.15/openmoe_base_wikitext-2-v1_language_modeling.pth")
    # booster.save_model(model, "/home/nus-hx/code/ColossalAI/examples/language/openmoe/outputs/2023.12.11-13.11.06/openmoe_base_yizhongw_super_natural_instruction_generation.pth", shard=True, size_per_shard=1)
    model = model.eval().bfloat16()
    model = model.to(torch.cuda.current_device())
    tokenizer = T5Tokenizer.from_pretrained("google/umt5-small")
    
    # input_str = """In this task, you need to count the number of nouns/verbs in the given sentence. Input: Sentence: 'A baseball player breaks a baseball bat while hitting the ball'. Count the number of nouns in this sentence. Output:"""
    input_str = """Homarus gammarus , known as the European lobster or common lobster , is a species of <unk> lobster from the eastern Atlantic Ocean , Mediterranean Sea and parts of the Black Sea . It is closely related to the American lobster , H. americanus . It may grow to a length of 60 cm ( 24 in ) and a mass of 6 kilograms ( 13 lb ) , and bears a conspicuous pair of claws . In life , the lobsters are blue , only becoming " lobster red " on cooking . Mating occurs in the summer , producing eggs which are carried by the females for up to a year before hatching into <unk> larvae ."""

    input_ids = tokenizer(
        "<pad>" + input_str,
        truncation=True,
        return_tensors="pt",
        add_special_tokens=False
    )
    # input_ids = tokenizer("<pad>" + input_str, return_tensors="pt", add_special_tokens=False)
    input_ids = input_ids.input_ids.to(torch.cuda.current_device())
    generation_output = model.module.generate(input_ids, use_cache=True, do_sample=True, max_new_tokens=64)
    out = tokenizer.decode(generation_output[0], skip_special_tokens=False)
    print(f"rank{rank} output: \n{out}\n")

def _run_dist(rank, world_size, args, port, parallel):
    colossalai.launch(
        config=dict(),
        rank=rank,
        world_size=world_size,
        host="localhost",
        port=port,
        backend="nccl",
    )
    inference_single(args, rank, parallel)

@rerun_if_address_is_in_use()
def infer_parallel(world_size, args, parallel):
    spawn(_run_dist, world_size, args=args, parallel=parallel)


if __name__ == "__main__":
    args = parse_args()
    if args.num_gpus <= 1:
        inference(args) # 1 GPU
    else:
        infer_parallel(world_size=args.num_gpus, args=args, parallel="ep_zero") # multi-gpus
