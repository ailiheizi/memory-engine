"""性格层: 多 LoRA adapter 隔离 + 可切换。

设计要点(规避灾难性遗忘):
    - 每个性格 = 一个独立 LoRA adapter (物理隔离)
    - 切换性格 = 卸载/加载 adapter (秒级)
    - 增: 训练新 adapter; 删: 删 adapter 目录; 改: 重训该 adapter
    - 性格之间互不污染 (不在同一组权重里编辑)

方案 A 用法: 带当前性格的小模型先生成一段"风格示范", 作为 few-shot
注入 DeepSeek 的 prompt, 让 DeepSeek 模仿该风格。
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PersonaManager:
    """管理多个性格 adapter: 训练、切换、生成风格示范。"""

    def __init__(self, base_model: str = "Qwen/Qwen3-0.6B", personas_dir: str = "personas"):
        self.base_model_id = base_model
        self.personas_dir = Path(personas_dir)
        self.personas_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.personas_dir / "registry.json"

        self._model = None       # 基础模型(共享)
        self._tokenizer = None
        self._active_persona: Optional[str] = None
        self._active_peft = None  # 当前加载的 PeftModel
        self.registry: dict = {}  # {persona_id: {"desc", "n_examples", "ts"}}
        self._load_registry()

    def _load_registry(self):
        if self.registry_file.exists():
            self.registry = json.loads(self.registry_file.read_text(encoding="utf-8"))

    def _save_registry(self):
        self.registry_file.write_text(json.dumps(self.registry, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 基础模型延迟加载 ----

    def _ensure_base(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info(f"Loading base model: {self.base_model_id}")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._model, self._tokenizer

    def _persona_path(self, persona_id: str) -> Path:
        return self.personas_dir / persona_id

    # ---- 增: 训练新性格 ----

    def create_persona(self, persona_id: str, examples: list[dict], desc: str = "", epochs: int = 10) -> dict:
        """训练一个新性格 adapter。

        examples: [{"user": "...", "response": "..."}] 风格示范对
        """
        import torch
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
        from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

        model, tokenizer = self._ensure_base()

        # 包 LoRA(从干净的 base 开始, 不复用已加载的 adapter)
        from peft import get_peft_model as _gpm
        peft_model = _gpm(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
        ))

        # 构造训练数据(只在 assistant 回复上算 loss)
        def tok(ex):
            msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": ex["user"]},
                {"role": "assistant", "content": ex["response"]},
            ]
            full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prompt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            fids = tokenizer(full, truncation=True, max_length=512)["input_ids"]
            pids = tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
            labels = [-100] * len(pids) + fids[len(pids):]
            return {"input_ids": fids, "attention_mask": [1] * len(fids), "labels": labels[:len(fids)]}

        ds = Dataset.from_list(examples).map(tok, remove_columns=Dataset.from_list(examples).column_names)
        args = TrainingArguments(
            output_dir=str(self._persona_path(persona_id) / "_train_tmp"),
            num_train_epochs=epochs, per_device_train_batch_size=1, gradient_accumulation_steps=1,
            learning_rate=1.5e-4, warmup_steps=2, save_strategy="no", report_to="none",
            use_cpu=True, optim="adamw_torch", logging_steps=50,
        )
        t0 = time.time()
        Trainer(model=peft_model, args=args, train_dataset=ds,
                data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")).train()
        elapsed = time.time() - t0

        # 保存 adapter
        save_path = self._persona_path(persona_id)
        peft_model.save_pretrained(str(save_path))

        # 卸载, 恢复 base(避免污染后续训练)
        peft_model = peft_model.unload()
        del peft_model

        self.registry[persona_id] = {"desc": desc, "n_examples": len(examples), "ts": int(time.time())}
        self._save_registry()
        logger.info(f"Created persona '{persona_id}' in {elapsed:.0f}s")
        return {"persona_id": persona_id, "train_time_s": elapsed, "n_examples": len(examples)}

    # ---- 删 / 列 ----

    def delete_persona(self, persona_id: str) -> bool:
        import shutil
        path = self._persona_path(persona_id)
        if path.exists():
            shutil.rmtree(path)
        if persona_id in self.registry:
            del self.registry[persona_id]
            self._save_registry()
        if self._active_persona == persona_id:
            self._active_persona = None
            self._active_peft = None
        logger.info(f"Deleted persona '{persona_id}'")
        return True

    def list_personas(self) -> dict:
        return dict(self.registry)

    # ---- 切换 ----

    def activate(self, persona_id: Optional[str]):
        """切换当前性格。persona_id=None 表示用裸 base(无性格)。"""
        if persona_id is None:
            self._active_persona = None
            self._active_peft = None
            return
        if persona_id not in self.registry:
            raise ValueError(f"Unknown persona: {persona_id}")
        from peft import PeftModel
        model, _ = self._ensure_base()
        self._active_peft = PeftModel.from_pretrained(model, str(self._persona_path(persona_id)))
        self._active_persona = persona_id
        logger.info(f"Activated persona '{persona_id}'")

    # ---- 方案A: 生成风格示范 ----

    def style_demo(self, user_msg: str, max_new_tokens: int = 80) -> str:
        """用当前性格的小模型生成一段风格示范回答。

        若无激活性格, 返回空串(降级为无性格)。
        """
        if self._active_peft is None:
            return ""
        import torch, re
        _, tokenizer = self._ensure_base()
        msgs = [
            {"role": "system", "content": "Respond naturally in your characteristic style."},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self._active_peft.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=0.5,
                do_sample=True, pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.1,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
