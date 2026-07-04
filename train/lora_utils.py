# train/lora_utils.py
import os
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


def parse_target_modules(s: str) -> List[str]:
    if s is None or len(str(s).strip()) == 0:
        return ["q_proj", "v_proj"]
    return [x.strip() for x in str(s).split(",") if x.strip()]


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True


def count_trainable_parameters(module: nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _get_by_path(obj: nn.Module, path: str) -> nn.Module:
    cur = obj
    for name in path.split("."):
        cur = getattr(cur, name)
    return cur


def _set_by_path(obj: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    cur = obj
    for name in parts[:-1]:
        cur = getattr(cur, name)
    setattr(cur, parts[-1], value)


def attach_lora_to_openfly_model(model: nn.Module, args):
    """
    Attach PEFT LoRA to the LLM submodule of OpenFly/OpenVLA.

    Important:
    We try submodules first instead of wrapping the root model first.
    This keeps custom OpenFly methods such as predict_action available.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError("peft is not installed. Run: pip install peft") from e

    freeze_module(model)

    target_modules = parse_target_modules(args.lora_target_modules)

    lora_config = LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        target_modules=target_modules,
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
    )

    candidate_paths = [
        "llm_backbone.llm",
        "language_model",
        "text_model",
        "model.language_model",
        "model",
        "__root__",
    ]

    last_error: Optional[Exception] = None

    for path in candidate_paths:
        try:
            if path == "__root__":
                peft_model = get_peft_model(model, lora_config)
                trainable, total = count_trainable_parameters(peft_model)
                print("[LoRA] attached to root model")
                print(
                    f"[LoRA] trainable={trainable} total={total} "
                    f"ratio={100.0 * trainable / max(total, 1):.6f}%"
                )
                try:
                    peft_model.print_trainable_parameters()
                except Exception:
                    pass
                return peft_model, "__root__"

            sub = _get_by_path(model, path)
            peft_sub = get_peft_model(sub, lora_config)
            _set_by_path(model, path, peft_sub)

            trainable, total = count_trainable_parameters(model)
            print(f"[LoRA] attached to submodule: {path}")
            print(
                f"[LoRA] trainable={trainable} total={total} "
                f"ratio={100.0 * trainable / max(total, 1):.6f}%"
            )
            try:
                peft_sub.print_trainable_parameters()
            except Exception:
                pass
            return model, path

        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(
        "LoRA attach failed. Check whether the model contains q_proj/v_proj. "
        f"Last error: {repr(last_error)}"
    )


def get_lora_root(model: nn.Module, lora_target_path: str) -> nn.Module:
    if lora_target_path == "__root__":
        return model
    return _get_by_path(model, lora_target_path)


def save_lora_adapter(
    model: nn.Module,
    lora_target_path: str,
    output_dir: str,
    tag: str = "last",
) -> str:
    adapter_dir = os.path.join(output_dir, f"lora_adapter_{tag}")
    os.makedirs(adapter_dir, exist_ok=True)

    lora_root = get_lora_root(model, lora_target_path)
    if not hasattr(lora_root, "save_pretrained"):
        raise RuntimeError("LoRA root does not support save_pretrained.")

    print(f"[LoRA] saving adapter to: {adapter_dir}")
    lora_root.save_pretrained(adapter_dir)

    meta_path = os.path.join(output_dir, "lora_target_path.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(lora_target_path)

    tag_meta_path = os.path.join(output_dir, f"lora_target_path_{tag}.txt")
    with open(tag_meta_path, "w", encoding="utf-8") as f:
        f.write(lora_target_path)

    return adapter_dir


def load_lora_adapter_for_eval(
    model: nn.Module,
    adapter_dir: str,
    lora_target_path: Optional[str] = None,
):
    if adapter_dir is None or len(str(adapter_dir).strip()) == 0:
        return model

    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_dir}")

    try:
        from peft import PeftModel
    except ImportError as e:
        raise ImportError("peft is not installed. Run: pip install peft") from e

    if lora_target_path is None:
        parent = os.path.dirname(adapter_dir)
        candidates = [
            os.path.join(parent, "lora_target_path.txt"),
            os.path.join(parent, "lora_target_path_best.txt"),
            os.path.join(parent, "lora_target_path_last.txt"),
        ]
        for meta_path in candidates:
            if os.path.isfile(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    lora_target_path = f.read().strip()
                break

        if not lora_target_path:
            lora_target_path = "__root__"

    print(f"[LoRA] loading adapter from: {adapter_dir}")
    print(f"[LoRA] target path: {lora_target_path}")

    if lora_target_path == "__root__":
        return PeftModel.from_pretrained(model, adapter_dir)

    sub = _get_by_path(model, lora_target_path)
    sub = PeftModel.from_pretrained(sub, adapter_dir)
    _set_by_path(model, lora_target_path, sub)
    return model


def build_lora_optimizer(
    model: nn.Module,
    extra_modules: Iterable[nn.Module],
    args,
):
    lora_params = []
    head_params = []

    for _, p in model.named_parameters():
        if p.requires_grad:
            lora_params.append(p)

    for module in extra_modules:
        if module is None:
            continue
        for p in module.parameters():
            if p.requires_grad:
                head_params.append(p)

    param_groups = []

    if len(lora_params) > 0:
        param_groups.append(
            {
                "params": lora_params,
                "lr": float(args.learning_rate),
                "weight_decay": float(args.weight_decay),
            }
        )

    if len(head_params) > 0:
        param_groups.append(
            {
                "params": head_params,
                "lr": float(args.head_learning_rate),
                "weight_decay": float(args.weight_decay),
            }
        )

    if len(param_groups) == 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "Check LoRA adapter and dual head requires_grad."
        )

    print(f"[Optimizer] LoRA tensors: {len(lora_params)}")
    print(f"[Optimizer] Head tensors: {len(head_params)}")

    return torch.optim.AdamW(param_groups)


def trainable_parameters(model: nn.Module, extra_modules: Iterable[nn.Module]):
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    for module in extra_modules:
        if module is None:
            continue
        for p in module.parameters():
            if p.requires_grad:
                params.append(p)

    return params
