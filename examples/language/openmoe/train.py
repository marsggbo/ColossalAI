import argparse
import os
from functools import partial
from typing import Dict

import torch
import torch.distributed as dist
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import snapshot_download
from model.modeling_openmoe import OpenMoeForCausalLM, set_openmoe_args
from model.openmoe_policy import OpenMoeForCausalLMPolicy
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import T5Tokenizer
from transformers.models.llama import LlamaConfig

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.moe.layers import apply_load_balance
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.utils import skip_init
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def load_ckpt(repo_name: str, model: OpenMoeForCausalLM, booster: Booster):
    ckpt_path = snapshot_download(repo_name)
    # single ckpt
    if os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")):
        ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin")
    # shard ckpt
    elif os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin.index.json")):
        ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin.index.json")
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
    booster.load_model(model, ckpt_path, strict=False)


def tokenize_data(batch, args, tokenizer: T5Tokenizer, max_length: int) -> Dict:
    if args.dataset == 'yizhongw/self_instruct':
        texts = ["<pad>" + sample["prompt"] + sample["completion"] for sample in batch] # `--dataset yizhongw/self_instruct --task_name super_natural_instructions``
    elif args.dataset == 'wikitext':
        texts = ["<pad>"+sample['text'] for sample in batch] # `--dataset wikitext --task_name wikitext-2-v1`
    data = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    data = {k: v.cuda() for k, v in data.items()}
    data["labels"] = data["input_ids"].clone()
    return data


class RandomDataset(Dataset):
    def __init__(self, num_samples: int = 1000, max_length: int = 2048, vocab_size: int = 32000, tokenizer=None):
        self.num_samples = num_samples
        self.max_length = max_length
        self.input_ids = torch.randint(0, vocab_size, (num_samples, max_length), device=get_current_device())
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }


def parse_args():
    # basic settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="base",
        choices=["base", "8b", "test"],
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--plugin",
        type=str,
        default="hybrid",
        choices=["ep", "ep_zero", "hybrid"],
        help="Parallel methos. ep_zero is recommended for general cases. ep can provides least memory consumption and hybrid suits large scale training.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs",
        help="The path of your saved model after finetuning.",
    )
    parser.add_argument("--num_epoch", type=int, default=1, help="Number of epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per dp group) for the training dataloader.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help=" The interval (steps) of saving checkpoints.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="The mixed precision training.",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="yizhongw/self_instruct",
        help="dataset name from `datasets` repo.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="super_natural_instructions",
        help="task of corresponding dataset.",
    )

    # optim
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay to use.")

    # zero stage for all plugins
    parser.add_argument("--zero_stage", type=int, default=2, help="zero stage.")
    # ep_zero plugin
    parser.add_argument(
        "--extra_dp_size", type=int, default=1, help="ep_zero plugin's moe dp size. Recommended to be 2 or 4."
    )
    # hybrid plugin
    parser.add_argument("--pp_size", type=int, default=2, help="pp size for hybrid plugin")
    parser.add_argument("--dp_size", type=int, default=1, help="dp size for hybrid plugin")
    parser.add_argument("--ep_size", type=int, default=2, help="ep size for hybrid plugin")
    parser.add_argument("--microbatch_size", type=int, default=1, help="Microbatch size in pipeline for hybrid plugin")

    # kernel
    parser.add_argument(
        "--use_kernel",
        action="store_true",
        help="Use kernel optim. Need to install flash attention and triton to enable all kernel optimizations. Skip if not installed.",
    )
    parser.add_argument(
        "--use_layernorm_kernel",
        action="store_true",
        help="Use layernorm kernel. Need to install apex. Raise error if not installed.",
    )

    # loss
    parser.add_argument(
        "--router_aux_loss_factor",
        type=float,
        default=0.01,
        help="Moe router z loss. You can refer to STMoE for details.",
    )
    parser.add_argument(
        "--router_z_loss_factor",
        type=float,
        default=0.0001,
        help="Moe router aux loss. You can refer to STMoE for details.",
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing.")
    parser.add_argument(
        "--z_loss_factor", type=float, default=0.0001, help="The final outputs' classification z loss factor."
    )

    # load balance
    parser.add_argument(
        "--load_balance", action="store_true", help="Expert load balance. Defaults to False. Recommend to enable."
    )
    parser.add_argument("--load_balance_interval", type=int, default=1000, help="Expert load balance interval.")
    # communicate overlap
    parser.add_argument(
        "--comm_overlap",
        action="store_true",
        help="Use communication overlap for MoE. Recommended to enable for muiti-node training.",
    )
    # hierarchical all-to-all
    parser.add_argument(
        "--hierarchical_alltoall",
        action="store_true",
        help="Use hierarchical all-to-all for MoE. Recommended to enable for muiti-node training.",
    )
    
    # debug
    parser.add_argument(
        "--ipdb",
        action="store_true",
        help="enable debug mode using ipdb",
    )
    # test mode
    parser.add_argument(
        "--valid_only",
        action="store_true",
        help="Only run the test code.",
    )
    parser.add_argument(
        "--valid_ckpt_path",
        type=str,
        help="The checkpoint path for validation.",
    )
    parser.add_argument(
        "--comment",
        type=str,
        help="Comment for the experiment.",
    )

    args = parser.parse_args()
    return args


def perplexity_func(logits, target_ids):
    # 获取有效的标签数量，因为模型内部向左偏移了一个位置
    trg_len = target_ids.size(1)
    valid_labels = target_ids[:, 1:]  # 剔除起始标记 [CLS]，获取有效标签
    valid_logits = logits[:, :-1, :]  # 对 logits 进行相应裁剪，使其与有效标签对齐

    # 定义损失函数为交叉熵损失
    loss_function = torch.nn.CrossEntropyLoss()

    # 将 logits 展平为二维张量，将有效标签展平为一维张量
    logits_flat = valid_logits.contiguous().view(-1, valid_logits.size(-1))
    labels_flat = valid_labels.contiguous().view(-1)

    # 计算交叉熵损失
    neg_log_likelihood = loss_function(logits_flat, labels_flat)
    ppl = torch.exp(neg_log_likelihood).item()
    return ppl


def validate(args, dataloader, model, tokenizer):
    model.eval()
    ppls = []
    device = torch.cuda.current_device()
    local_rank = dist.get_rank()
    dataloader_iter = iter(dataloader)
    total_len = len(dataloader_iter)
    with torch.no_grad():
        for idx in range(total_len):
            data = next(dataloader_iter)
            data = move_to_cuda(data, device)
            input_ids = data["input_ids"]
            if idx < 3:
                print(f"rank {local_rank}: {input_ids[:2, :10]}")
            output = model(input_ids, return_dict=True)
            logits = output.logits
            ppl = perplexity_func(logits, input_ids)
            ppls.append(ppl)
            if idx % 10 == 0:
                print(f"rank {local_rank} batch {idx}: {torch.tensor(ppls).mean()}")

    ppl = torch.tensor(ppls).mean()
    print(f"rank {local_rank}: test ppl={ppl:.4f}")
    return ppl


def main():
    args = parse_args()
    if args.ipdb:
        from ipdb import set_trace
        set_trace()

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()
    test_mode = args.model_name == "test"

    # Set plugin
    booster_kwargs = {}
    hybrid_dict = {
        "tp_size": 1,
        "custom_policy": OpenMoeForCausalLMPolicy(),
        "enable_fused_normalization": args.use_layernorm_kernel,
        "enable_jit_fused": args.use_kernel,
        "precision": args.precision,
        "zero_stage": args.zero_stage,
    }
    mgr_dict = {}
    if args.plugin == "ep":
        dp_size = dist.get_world_size()
        plugin = MoeHybridParallelPlugin(
            pp_size=1,
            **hybrid_dict,
        )
        MOE_MANAGER.setup(
            parallel="EP",
            max_ep_size=dp_size,
            **mgr_dict,
        )
    elif args.plugin == "ep_zero":
        dp_size = dist.get_world_size()
        use_ep_inside = False
        plugin = MoeHybridParallelPlugin(
            pp_size=1,
            extra_dp_size=args.extra_dp_size,
            use_ep_inside=use_ep_inside,
            **hybrid_dict,
        )
        MOE_MANAGER.setup(
            parallel="EP",
            max_ep_size=dp_size // args.extra_dp_size,
            use_ep_inside=use_ep_inside,
            **mgr_dict,
        )
    elif args.plugin == "hybrid":
        dp_size = dist.get_world_size() // args.pp_size
        plugin = MoeHybridParallelPlugin(
            pp_size=args.pp_size,
            microbatch_size=args.microbatch_size,
            **hybrid_dict,
        )
        MOE_MANAGER.setup(
            parallel="EP",
            mode="fixed",
            fixed_dp_size=args.dp_size,
            fixed_ep_size=args.ep_size,
            fixed_pp_size=args.pp_size,
            **mgr_dict,
        )
    else:
        raise ValueError(f"Invalid plugin {args.plugin}")
    coordinator.print_on_master(f"Set plugin as {plugin.__class__.__name__}")

    # Build OpenMoe model
    if test_mode:
        config = LlamaConfig.from_pretrained("hpcaitech/openmoe-base")
        config.hidden_size = 128
        config.intermediate_size = 256
        config.vocab_size = 32000
    else:
        repo_name = "hpcaitech/openmoe-" + args.model_name
        config = LlamaConfig.from_pretrained(repo_name)
    set_openmoe_args(
        config,
        num_experts=config.num_experts,
        moe_layer_interval=config.moe_layer_interval,
        router_aux_loss_factor=args.router_aux_loss_factor,
        router_z_loss_factor=args.router_z_loss_factor,
        z_loss_factor=args.z_loss_factor,
        enable_load_balance=args.load_balance,
        enable_comm_overlap=args.comm_overlap,
        enable_hierarchical_alltoall=args.hierarchical_alltoall,
        enable_kernel=args.use_kernel,
    )
    print(config)
    print(args)
    with skip_init():
        model = OpenMoeForCausalLM(config)
    params = sum([p.numel() for p in model.parameters()])
    coordinator.print_on_master(f"Finish init model with config:\n{config} #params={params}")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare tokenizer and dataloader
    tokenizer = T5Tokenizer.from_pretrained("google/umt5-small")
    if test_mode:
        dataset = RandomDataset(num_samples=20, tokenizer=tokenizer)
        collate_fn = None
    else:
        dataset = load_dataset(args.dataset, args.task_name)
        collate_fn = partial(tokenize_data, args=args, tokenizer=tokenizer, max_length=args.max_length)
    if 'validation' in dataset:
        train_valid_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
        train_dataloader_local = plugin.prepare_dataloader(
            train_valid_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn
        )
    else:
        train_dataloader_local = plugin.prepare_dataloader(
            dataset["train"], batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn
        )
    test_dataloader_local = plugin.prepare_dataloader(
        dataset["test"], batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn
    )

    # Set optimizer
    optimizer = HybridAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    if not test_mode:
        load_ckpt(repo_name, model, booster)
    if args.valid_only:
        assert args.valid_ckpt_path is not None, "please specify the checkpoint file(*.pth) path via --valid_ckpt_path"
        valid_ckpt_path = args.valid_ckpt_path
        assert valid_ckpt_path is not None, "Please specify the checkpoint file(*.pth) path"
        ckpt = torch.load(valid_ckpt_path)
        state_dict = {}
        for key, value in ckpt.items():
            if key.startswith("module."):
                state_dict[key[7:]] = value
            else:
                state_dict[key] = value
        model.load_state_dict(state_dict)
        model, optimizer, _, test_dataloader, _ = booster.boost(model=model, optimizer=optimizer, dataloader=test_dataloader_local)
    else:
        model, optimizer, _, train_dataloader, _ = booster.boost(model=model, optimizer=optimizer, dataloader=train_dataloader_local)
    use_pipeline = isinstance(booster.plugin, MoeHybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    coordinator.print_on_master(f"Finish init booster")

    if args.valid_only:
        coordinator.print_on_master(f"Start testing")
        validate(args, test_dataloader, model, tokenizer)
        coordinator.print_on_master(f"Finish testing")
        return

    # Start finetuning
    for epoch in range(args.num_epoch):
        model.train()
        train_dataloader_iter = iter(train_dataloader)
        total_len = len(train_dataloader_iter)
        with tqdm(
            range(total_len),
            desc=f"Epoch [{epoch + 1}/{args.num_epoch}]",
            disable=not coordinator.is_master(),
        ) as pbar:
            for step in pbar:
                if use_pipeline:
                    # Forward pass
                    outputs = booster.execute_pipeline(
                        train_dataloader_iter,
                        model,
                        lambda x, y: x.loss,
                        optimizer,
                        return_loss=True,
                        return_outputs=True,
                    )
                    # Backward and optimize
                    if is_pp_last_stage:
                        loss = outputs["loss"]
                        pbar.set_postfix({"loss": loss.item()})
                else:
                    # Forward pass
                    data = next(train_dataloader_iter)
                    data = move_to_cuda(data, torch.cuda.current_device())
                    outputs = model(**data)
                    loss = outputs["loss"]
                    # Backward
                    booster.backward(loss, optimizer)
                    pbar.set_postfix({"loss": loss.item()})

                optimizer.step()
                optimizer.zero_grad()

                # Apply load balance
                if (
                    args.load_balance
                    and args.load_balance_interval > 0
                    and (step + 1) % args.load_balance_interval == 0
                ):
                    coordinator.print_on_master(f"Apply load balance")
                    apply_load_balance(model, optimizer)
                # save ckeckpoint
                if (step + 1) % args.save_interval == 0:
                    coordinator.print_on_master(f"Saving model checkpoint to {args.output_path}")
                    booster.save_model(model, args.output_path, shard=True)

        # save checkpoint at the end of each epochs
        booster.save_model(model, args.output_path, shard=True)
        coordinator.print_on_master(f"Saving model checkpoint to {args.output_path}")

    # Finish training
    coordinator.print_on_master(f"Finish training")


if __name__ == "__main__":
    main()
