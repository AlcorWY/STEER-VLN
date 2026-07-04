from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch


def _parse_target_modules(value) -> List[str]:
    if value is None:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
        return items if items else ["q_proj", "k_proj", "v_proj", "o_proj"]

    if isinstance(value, (list, tuple)):
        items = [str(x).strip() for x in value if str(x).strip()]
        return items if items else ["q_proj", "k_proj", "v_proj", "o_proj"]

    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def _get_module_by_path(model, path: str):
    if path in ["", ".", "root", None]:
        return model

    cur = model
    for part in str(path).split("."):
        if not part:
            continue
        if not hasattr(cur, part):
            raise AttributeError(f"Module path not found: {path}, missing: {part}")
        cur = getattr(cur, part)

    return cur


def _set_module_by_path(model, path: str, new_module):
    if path in ["", ".", "root", None]:
        return new_module

    parts = str(path).split(".")
    cur = model

    for part in parts[:-1]:
        if not hasattr(cur, part):
            raise AttributeError(f"Module path not found: {path}, missing: {part}")
        cur = getattr(cur, part)

    setattr(cur, parts[-1], new_module)
    return model


def _candidate_lora_target_paths(model) -> List[str]:
    candidates = [
        "llm_backbone.llm",
        "language_model",
        "model.language_model",
        "model",
        "root",
    ]

    valid = []

    for path in candidates:
        try:
            _get_module_by_path(model, path)
            valid.append(path)
        except Exception:
            pass

    if "root" not in valid:
        valid.append("root")

    return valid


def unfreeze_module(module):
    module.train()
    for p in module.parameters():
        p.requires_grad_(True)
    return module


def freeze_module(module):
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def attach_lora_to_openfly_model(model, args) -> Tuple[torch.nn.Module, str]:
    """
    Attach PEFT LoRA to the language backbone inside OpenFly.

    Returns:
        model: model with LoRA attached
        lora_target_path: module path where adapter is attached

    Default target priority:
        llm_backbone.llm -> language_model -> model.language_model -> model -> root
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as e:
        raise ImportError("peft is required for STEER_VLN LoRA training. Please install peft.") from e

    # Freeze everything first; PEFT will unfreeze LoRA adapter parameters.
    for p in model.parameters():
        p.requires_grad_(False)

    rank = int(getattr(args, "lora_rank", 8))
    alpha = int(getattr(args, "lora_alpha", 16))
    dropout = float(getattr(args, "lora_dropout", 0.05))
    target_modules = _parse_target_modules(getattr(args, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj"))

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    candidate_paths = _candidate_lora_target_paths(model)
    errors = []

    for path in candidate_paths:
        try:
            target = _get_module_by_path(model, path)
            peft_target = get_peft_model(target, lora_cfg)

            if path == "root":
                model = peft_target
            else:
                model = _set_module_by_path(model, path, peft_target)

            print(
                f"[LoRA] attached to {path}; "
                f"r={rank}, alpha={alpha}, dropout={dropout}, target_modules={target_modules}",
                flush=True,
            )

            try:
                peft_target.print_trainable_parameters()
            except Exception:
                pass

            return model, path

        except Exception as e:
            errors.append((path, repr(e)))

    msg = "\n".join([f"  {p}: {err}" for p, err in errors])
    raise RuntimeError(f"Failed to attach LoRA to OpenFly model. Tried:\n{msg}")


def _trainable_params(module) -> List[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad]


def build_lora_optimizer(model, extra_modules: Optional[Iterable[torch.nn.Module]], args):
    """
    Optimizer for:
      - LoRA adapter params in OpenFly
      - TH-S head params
    """
    lr = float(getattr(args, "learning_rate", 3e-5))
    head_lr = float(getattr(args, "head_learning_rate", lr))
    weight_decay = float(getattr(args, "weight_decay", 1e-4))

    groups = []

    lora_params = _trainable_params(model)
    if lora_params:
        groups.append(
            {
                "params": lora_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "lora",
            }
        )

    if extra_modules:
        for i, module in enumerate(extra_modules):
            params = _trainable_params(module)
            if params:
                groups.append(
                    {
                        "params": params,
                        "lr": head_lr,
                        "weight_decay": weight_decay,
                        "name": f"extra_{i}",
                    }
                )

    if not groups:
        raise RuntimeError("No trainable parameters found for STEER_VLN LoRA optimizer.")

    print("[Optimizer] parameter groups:", flush=True)
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(f"  - {g.get('name', 'group')}: params={n}, lr={g['lr']}, wd={g['weight_decay']}", flush=True)

    return torch.optim.AdamW(groups)


def trainable_parameters(model, extra_modules: Optional[Iterable[torch.nn.Module]] = None):
    model_total = sum(p.numel() for p in model.parameters())
    model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    extra_total = 0
    extra_trainable = 0

    if extra_modules:
        for m in extra_modules:
            extra_total += sum(p.numel() for p in m.parameters())
            extra_trainable += sum(p.numel() for p in m.parameters() if p.requires_grad)

    total = model_total + extra_total
    trainable = model_trainable + extra_trainable
    ratio = trainable / max(total, 1)

    return {
        "model_trainable": int(model_trainable),
        "model_total": int(model_total),
        "extra_trainable": int(extra_trainable),
        "extra_total": int(extra_total),
        "trainable": int(trainable),
        "total": int(total),
        "ratio": float(ratio),
    }


def _find_peft_target(model, lora_target_path: Optional[str]):
    if lora_target_path:
        return _get_module_by_path(model, lora_target_path), lora_target_path

    # Prefer known paths. Use the first module that has PEFT methods/configs.
    for path in _candidate_lora_target_paths(model):
        try:
            target = _get_module_by_path(model, path)
            if hasattr(target, "save_pretrained") and (
                hasattr(target, "peft_config") or target.__class__.__name__.lower().startswith("peft")
            ):
                return target, path
        except Exception:
            pass

    # Fallback root.
    return model, "root"


def save_lora_adapter(model, lora_target_path: str, output_dir: str, tag: str):
    """
    Save LoRA adapter to:
      output_dir/lora_adapter_{tag}
    """
    output_dir = Path(output_dir)
    adapter_dir = output_dir / f"lora_adapter_{tag}"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    target, resolved_path = _find_peft_target(model, lora_target_path)

    if not hasattr(target, "save_pretrained"):
        raise AttributeError(
            f"LoRA target has no save_pretrained(): path={resolved_path}, type={type(target)}"
        )

    target.save_pretrained(str(adapter_dir))

    with open(adapter_dir / "lora_target_path.txt", "w", encoding="utf-8") as f:
        f.write(str(resolved_path) + "\n")

    print(f"[LoRA] saved adapter: {adapter_dir}", flush=True)
    print(f"[LoRA] target path: {resolved_path}", flush=True)

    return str(adapter_dir)


def _read_saved_target_path(adapter_dir: Path) -> Optional[str]:
    p = adapter_dir / "lora_target_path.txt"
    if p.exists():
        value = p.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def load_lora_adapter_for_eval(model, adapter_dir: str, lora_target_path: str = None):
    """
    Load PEFT LoRA adapter for evaluation.

    If lora_target_path is not supplied, this function tries:
      1. adapter_dir/lora_target_path.txt
      2. known OpenFly language-backbone paths
      3. root model
    """
    try:
        from peft import PeftModel
    except Exception as e:
        raise ImportError("peft is required for loading LoRA adapters. Please install peft.") from e

    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_dir}")

    saved_path = _read_saved_target_path(adapter_dir)
    if not lora_target_path and saved_path:
        lora_target_path = saved_path

    candidate_paths = [lora_target_path] if lora_target_path else _candidate_lora_target_paths(model)

    errors = []

    for path in candidate_paths:
        if not path:
            continue

        try:
            target = _get_module_by_path(model, path)
            peft_target = PeftModel.from_pretrained(
                target,
                str(adapter_dir),
                is_trainable=False,
            )

            if path == "root":
                model = peft_target
            else:
                model = _set_module_by_path(model, path, peft_target)

            model.eval()

            print(f"[LoRA Eval] loaded adapter: {adapter_dir}", flush=True)
            print(f"[LoRA Eval] target path: {path}", flush=True)

            return model

        except Exception as e:
            errors.append((path, repr(e)))

    msg = "\n".join([f"  {p}: {err}" for p, err in errors])
    raise RuntimeError(f"Failed to load LoRA adapter. Tried:\n{msg}")
