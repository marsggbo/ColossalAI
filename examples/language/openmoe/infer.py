from argparse import ArgumentParser
import time
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
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--inf_mode", default='sequential', type=str, help="inference mode")
    parser.add_argument("--num_gpus", default=1, type=int, help="number of GPUs for inference")
    return parser.parse_args()


##################################
#           1 GPU inference     #
##################################

def schedule(lengths, mini_batch_size=1, vbs=False):
    """
    根据给定的序列长度列表，生成训练批次。
    
    :param lengths: 列表，包含每个序列的长度。
    :param mini_batch_size: int, 迷你批次的大小。
    :param vbs: bool, 是否使用可变批次大小。
    :return: 返回一个列表，其中每个元素是一个元组，包含批次的ID列表和最大长度。
    """
    from collections import defaultdict
    def buckit(x, cell=50):
        x = int(x)
        x = (x // cell + 1) * cell if x % cell != 0 else x
        return x
    def len_bs_dict_fn(x):
        len_bs_dict = {
            50: 256,
            100: 128,
            150: 64,
            200: 32,
            250: 16,
            300: 16,
        }
        if x in len_bs_dict:
            bs = len_bs_dict[x]
        else:
            bs = 16
        return bs
    
    # 根据序列长度对序列ID进行排序
    lengths_with_id = [(i, l) for i, l in enumerate(lengths)]
    sorted_lengths_with_id = sorted(lengths_with_id, key=lambda x: x[1], reverse=False)

    batches = []
    
    # 固定批次大小
    if not vbs:
        for i in range(0, len(lengths), mini_batch_size):
            batch = sorted_lengths_with_id[i : i + mini_batch_size]
            batch_ids = [x[0] for x in batch]
            max_len = max([x[1] for x in batch])
            batches.append((batch_ids, max_len))
    
    # 使用可变批次大小
    else:
        # 根据序列长度进行分组
        ids_len_dict = defaultdict(list)
        for i, l in sorted_lengths_with_id:
            ids_len_dict[buckit(l)].append(i)
        
        # 根据可变批次大小进行批次划分
        max_l = max(lengths)
        for l, ids in ids_len_dict.items():
            bs = len_bs_dict_fn(l)
            for i in range(0, len(ids), bs):
                batch = ids[i : i + bs]
                if l < max_l and len(batch) < max(len_bs_dict_fn(l) // 2, 16):
                    l_ = l + 50
                    while l_ not in ids_len_dict:
                        l_ += 50
                    ids_len_dict[l_].extend(batch)
                    break
                batches.append((batch, l))
    
    return batches


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
            ckpt = torch.load(ckpt_path)
            state_dict = {}
            for key, value in ckpt.items():
                if key.startswith("module."):
                    state_dict[key[7:]] = value
                else:
                    state_dict[key] = value
            model.load_state_dict(state_dict)
        else:
            print("The checkpoint path is not specified.")
        # model = OpenMoeForCausalLM.from_pretrained(f"hpcaitech/openmoe-{args.model}", config=config)
    model = model.eval().bfloat16()
    model = model.to(torch.cuda.current_device())

    # # # input examples for yizhongw datasets
    prompts = [
        """In this task, you are given two sets, and a question. You need to find whether an element is at the intersection of two given sets. A Set is shown by two curly braces and comma-separated numbers inside, like {1, 2, 3}. The intersection of two given sets is the largest set which contains all the elements that are common to both sets. An element is at the intersection of two given sets, A and B, if common to both A and B. Classify your answers into 'Yes' or 'No'. Input: Set1: '{2, 6, 8, 9, 11, 13, 18}', Set2: '{2, 12}'. Is the element '9' in the intersection of Set1 and Set2 ? Output:""",
        """In this task you are given a question. You need to generate an answer to the question. Input: Question:What is the opposite side from starboard on a ship? Output:""",
        """In this task, you are given a list of integers and an integer k. You need to add integer k to each element in the list and return the updated list. Input: [133, 281, 294, 97, 66, 269, 39, 92, 25, 162, 16, 226, 76, 119, 268, 12, 296, 103, 211, 163] k=1. Output:""",
        """In this task, you are given a country name and you need to return the capital city of the given country. Input: Falkland Islands. Output:""",
        """In this task, the input is a set of dialogues between a user and an assistant. You need to find the dialogue that is basically a response given to a question or an aspect of the user. Input: Cool. Anything more about its history? I see... Anything more about their economy? In recent years, economic growth has been impressive, reaching a 6.9% in 2007, one of the highest economic growth rate in Latin America. Cool. Can you tell me about its history? That's very interesting. Can you tell me about their economy? Output:""",
        """In this task you are given a tweet that contains some form of irony. You must classify the type of irony the tweet has. Label the tweets ("polarity","situational","other") based on the irony they have. Situational irony happens when a situation fails to meet some expectations, Label these instances as "situational". polarity irony happens when irony is achieved by inverting the intended sentence, Label these instances as "polarity". There are other kinds of ironies that are neither polarity nor situational, Label these instances as "other". Note that URLs in the text have been replaced with [Link]. Input: Pulis is available. Will guarantee we stay up. Output:""",
        """In this task, you are given an input list A. You need to extract and sort the unique digits used in the list in ascending order. Return -1 if there is no digit in the list. Input: ['425', '29', 'i', '255', '201', 'r', '29', '191', '441', '239', 'o'] Output:""",
        """Given a math problem with context and a question and 5 answer choices, the task is to provide the correct answer choice based on the problem. You must choose one of the given answer choices by letter: a, b, c, d, or e; anything else is invalid. Input: Problem: a rectangular plot measuring 90 metres by 50 metres is to be enclosed by wire fencing. if the poles of the fence are kept 10 metres apart, how many poles will be needed ? Options: a. 28, b. 56, c. 57, d. 58, e. none of these. Output:""",
        """In this task, you are given a set of context paragraphs, some supporting facts and an answer of a question. Your task is to generate question for given answer based on set of context paragraphs, supporting facts and an answer. Input: Context_1 : Miss Seventeen is a reality television show on MTV that aired from October 17, 2005 to December 19, 2005. The show consisted of 17 young women competing for an internship at and a college scholarship. Atoosa Rubenstein was the main judge, she was the youngest editor-in-chief ever to run "Seventeen magazine". They picked 17 girls from around the United States who were not only photogenic but also had been at the top of their class, to provide a role model for young women. The girls were flown to New York, where they would take part in a contest similar in format to The Apprentice — they would be given tasks to be done by Atoosa, and in each episode one of the girls would be eliminated from the competition. The winner would get her face on the cover of "Seventeen magazine", a college scholarship and would be offered an internship job on the magazine. Context_2 : The Sancy, a pale yellow diamond of 55.23 carat, was once reputed to have belonged to the Mughals of antiquity, but is more likely of Indian origin owing to its cut, which is unusual by Western standards. Context_3 : The Spirit of de Grisogono is the world's largest cut black diamond and the world's fifth largest diamond overall. Starting at an uncut weight of 587 carat, it was taken from its origin in western Central African Republic and cut by Swiss jeweler De Grisogono. The resulting mogul-cut diamond weighs 312.24 carat and is set in a white gold ring with 702 smaller white diamonds totaling 36.69 carat. The ring is said to have been sold. Context_4 : Love & Letter, also known as First Love & Letter, is the first studio album by South Korean boy group Seventeen released on April 29, 2016. The album is a follow-up to the group's two EPs, "17 Carat" and "Boys Be" (2015). Context_5 : Rules of origin are used to determine the country of origin of a product for purposes of international trade. There are two common types of rules of origin depending upon application, the preferential and non-preferential rules of origin (19 CFR 102). The exact rules vary from country to country, from agreement to agreement. Context_6 : 17 Carat is the debut extended play by South Korean boy group Seventeen. It was released on May 29, 2015 by Pledis Entertainment and distributed by LOEN Entertainment. "Adore U" serves as the lead single for the extended play. Context_7 : Seventeen (Hangul: 세븐틴 ), also stylized as SEVENTEEN or SVT, is a South Korean boy group formed by Pledis Entertainment in 2015. The group consists of thirteen members who are separated into three sub-units, each with different areas of specialization: a 'Hip-Hop Unit', 'Vocal Unit', and 'Performance Unit'. They have released one studio album and four extended plays. Context_8 : "Fourteen Carat Mind" is a song written by Dallas Frazier and Larry Lee, and recorded by American country music artist Gene Watson. It was released in September 1981 as the first single from the album "Old Loves Never Die". "Fourteen Carat Mind" was Gene Watson's twentieth country hit and his only song to hit number one on the country chart. The single stayed at number one for one week and spent a total of fifteen weeks on the country chart. Context_9 : The Aurora Green Diamond is a 5.03 carat vivid green diamond with VS2 clarity. In May 2016, the Aurora Green became the largest ever vivid green diamond to ever sell at auction. The record was previous held by a 2.54 carat Fancy Vivid Green VS1 diamond that was sold by Sotheby’s on November 17, 2009 for $1.22 million per carat according to the Diamond Investment & Intelligence Center. On May 31, 2016, the diamond, which was originally owned by Scarselli Diamonds was sold by Christie's for a record price per carat of $3.3 million to Chinese jewelry company Chow Tai Fook, totaling $16.8 million. Context_10 : The South Korean boy band Seventeen embarked on their first concert tour entitled Seventeen 1st Asia Tour 2016 Shining Diamonds in July through September of 2016, performing at venues including Singapore, Australia, New Zealand and China. The string of concerts began in South Korea where 13,000 tickets were sold. They have also held four showcases, the most notable being their debut showcase, "Seventeen 1st Mini Album '17 Carat' Showcase""",
    ] * 5
    
    ### input examples for wikitext-2-v1
    # prompts = [
    #     """Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . <unk> the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " <unk> Raven " .""",
    #     """Troops are divided into five classes : Scouts , <unk> , Engineers , <unk> and Armored Soldier . <unk> can switch classes by changing their assigned weapon . Changing class does not greatly affect the stats gained while in a previous class . With victory in battle , experience points are awarded to the squad , which are distributed into five different attributes shared by the entire squad , a feature differing from early games ' method of distributing to different unit types .""",
    #     """The plain maskray generally hunts at the surface of the bottom substrate , rather than digging for prey . Its diet consists predominantly of <unk> shrimp and polychaete worms . Small bony fishes are also eaten , along with the occasional <unk> <unk> or <unk> . Larger rays consume a greater variety of prey and relatively more polychaete worms when compared to smaller rays . This species is <unk> by the <unk> <unk> <unk> .""",
    #     """For several years the arsenal , which was owned by the federal government , served as a simple arms depot and was staffed with only a handful of soldiers . But in November 1860 , with the American Civil War on the horizon , a company of the Second United States Artillery , consisting of sixty @-@ five men , was transferred to Little Rock under the command of Captain James Totten . On January 15 , 1861 , the state legislature decided to hold a referendum to determine if a state convention should be held to consider the issue of <unk> and to elect delegates to such a convention . It was planned for February 18 ; however , events at the arsenal , would not wait . On January 28 , then Governor Henry Massey Rector informed Captain Totten that he and his soldiers would be " permitted to remain in the possession of the Federal officers until the State , by authority of the people , shall have determined to <unk> their connection with the General Government , " Totten responded to this by telling the Governor that his orders came from the United States Government and began a desperate but ultimately futile dispatch of letters and <unk> asking for reinforcements , although rumors were widely spread that they were already coming . The first telegraph wire to span between Little Rock and Memphis had recently been completed . Local attorney John M Harrel was asked to compose the first telegraph dispatched from Arkansas 's capital . In his message , Harrel reported unconfirmed rumors that more federal troops had been sent to reinforce the Little Rock Arsenal .""",
    #     """The United States troops at the outposts of the western frontier of the state and in the Indian nation have all been recalled from winter quarters to reinforce the garrison at Fort Smith . The garrison at Fort Smith had been previously transferred to the United States Arsenal in this city ( Little Rock ) . The arsenal is one of the richest <unk> of military stores in the United States and is supposed to be the ultimate destination of the <unk> [ sic ] ordered from the frontier .""",
    #     """The Gambia women 's national football team represents the Gambia in international football competition . The team , however , has not competed in a match recognised by FIFA , the sport 's international governing body , despite that organised women 's football has been played in the country since 1998 . The Gambia has two youth teams , an under @-@ 17 side that has competed in FIFA U @-@ 17 Women 's World Cup qualifiers , and"""
    #     """hello, my name is marsggbo, nice to meet you! May I"""
    # ]
    # input_str = prompts[-1][:100]

    print("model config: ", model.config)
    device = torch.cuda.current_device()

    # ###### batch inference
    # refer to: https://github.com/huggingface/transformers/blob/890e790e16084e58a1ecb9329c98ec3e76c45994/tests/test_modeling_gpt2.py#L430
    if args.inf_mode == 'batch_inf':
        temperature = 0.5
        max_new_tokens = 128
        stream = True
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        
        # use different length sentences to test batching
        sentences_inp = ['<pad>' + x for x in prompts]
        input_length = torch.tensor([tokenizer(x, return_tensors="pt").input_ids.size(1) for x in sentences_inp])
        values, indices = input_length.sort()
        batch_index_maxlength = schedule(input_length, 2)
        
        def sample(last_token_logits, temperature):
            if temperature < 1e-4:
                token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            return token

        all_time = 0.
        for batch_idx, (indices, max_seq_length) in enumerate(batch_index_maxlength):
            sentences = [sentences_inp[idx] for idx in indices]
        
            inputs = tokenizer(sentences, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(device)
            l_input_ids = len(input_ids[0])
            output_ids = input_ids
            attention_mask = inputs.attention_mask.to(device)
            ending = [-1] * len(sentences)
            torch.manual_seed(0)

            past_info = None
            start = time.perf_counter()
            for i in range(max_new_tokens):
                # generation
                if i == 0 and past_info is None:
                    out = model(input_ids, use_cache=True, attention_mask=attention_mask)
                else:
                    out = model(
                        input_ids=token,
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                    )

                # sample
                last_token_logits = out.logits[:, -1]
                token = sample(last_token_logits, temperature)
                output_ids = torch.cat((output_ids, token), dim=1)

                # update attn & kv cache
                past_key_values = out.past_key_values
                attn_dtype = attention_mask.dtype
                extend_mask = torch.ones(len(token), 1, dtype=attn_dtype).to(device)
                attention_mask = torch.cat((attention_mask, extend_mask), dim=1)

                # ending detection
                num_ended = 0
                for j in range(len(sentences)):
                    if ending[j] == -1 and token[j] == tokenizer.eos_token_id:
                        ending[j] = i
                    if ending[j] != -1:
                        num_ended += 1
                if num_ended == len(sentences) and stream:
                    break
            end = time.perf_counter()
            all_time += (end - start)
            
            # collect results
            results = []
            for i in range(len(output_ids)):
                if ending[i] != -1:
                    output_ = output_ids[i][: l_input_ids + ending[i]]
                    is_finished = True
                else:
                    output_ = output_ids[i]
                    is_finished = False
                sentence_out = tokenizer.decode(output_, skip_special_tokens=True)
                print(f"{i} outputs: {sentence_out}")
                output = sentence_out[len(sentences[i]) :]

                num_input_tokens = len(input_ids[i])
                num_output_tokens = len(tokenizer(output).input_ids)
                num_total_tokens = num_input_tokens + num_output_tokens
                length = output_ids[i].shape[0] - l_input_ids + 1

                # return
                result = dict(
                    input=sentences[i],
                    output=output,
                    sentence=sentence_out,
                    num_input_tokens=num_input_tokens,
                    num_output_tokens=num_output_tokens,
                    num_total_tokens=num_total_tokens,
                    is_finished=is_finished,
                    length=length,
                )
                results.append(result)
        print(f"batch inference takes {all_time}s")


    # ###### sequential inference
    elif args.inf_mode == 'sequential':
        all_time = 0.
        for idx, input_str in enumerate(prompts):
            # print(f"Input: \n{input_str}\n")
            input_ids = tokenizer("<pad>" + input_str, return_tensors="pt", add_special_tokens=False)
            input_ids = input_ids.input_ids.to(device)
            # print(input_ids.numel())
            start = time.perf_counter()
            # outputs = model.generate(input_ids, use_cache=True, do_sample=True)
            outputs = model.generate(input_ids, use_cache=True, do_sample=True, max_new_tokens=128)
            end = time.perf_counter()
            cost_time = end-start
            all_time += cost_time
            out = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"{idx} prompt takes {cost_time}s\n{out}\n\n==========")
            # print(f"output: \n{outputs.shape}")
            # print(f"cost time: {cost_time}s")
        print(f"sequential inf: {all_time}s")
    else:
        raise NotImplementedError(args.inf_mode)

    # # batch prompts with similar sequence length
    # for input_str in prompts:
    #     print(f"Input: \n{input_str}\n")
    #     input_ids = tokenizer("<pad>" + input_str, return_tensors="pt", add_special_tokens=False)
    #     input_ids = input_ids.input_ids.to(torch.cuda.current_device())
    #     generation_output = model.generate(input_ids, use_cache=True, do_sample=True)
    #     # generation_output = model.generate(input_ids, use_cache=True, do_sample=True, max_new_tokens=64)
    #     out = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    #     print(f"output: \n{out}\n")


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
    if args.debug:
        from ipdb import set_trace
        set_trace()
    if args.num_gpus <= 1:
        inference(args) # 1 GPU
    else:
        infer_parallel(world_size=args.num_gpus, args=args, parallel="ep_zero") # multi-gpus
