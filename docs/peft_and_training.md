# PEFT & Training (`taktiny.peft` & `taktiny.takt`)

TakTiny supports Parameter-Efficient Fine-Tuning (PEFT) with LoRA and end-to-end training using `Takt`.

---

## 1. Parameter-Efficient Fine-Tuning (LoRA)

`LoraConfig` and `apply_peft` inject low-rank adaptation matrices into specified linear projections (`q_proj`, `v_proj`, `gate_proj`, `up_proj`):

```python
from taktiny.peft import LoraConfig, apply_peft

# Define LoRA Configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32.0,
    target_modules=["q_proj", "v_proj", "gate_proj", "up_proj"],
    lora_dropout=0.05,
)

# Apply LoRA adapter to model
peft_model = apply_peft(model, peft_config)
```

---

## 2. End-to-End Training Lifecycle (`Takt`)

`Takt` manages full-lifecycle training, dataset streaming, learning rate schedules, and checkpointing:

```python
from taktiny.takt import Takt, TrainingConfig, DatasetConfig

# Configure training hyperparameters
train_config = TrainingConfig(
    learning_rate=2e-5,
    warmup_steps=100,
    max_steps=1000,
    batch_size=8,
    weight_decay=0.01,
)

# Configure dataset parameters
dataset_config = DatasetConfig(
    dataset_name="wikitext",
    dataset_config_name="wikitext-2-raw-v1",
    seq_len=512,
)

# Instantiate Takt Trainer
trainer = Takt(
    model=peft_model,
    training_config=train_config,
    dataset_config=dataset_config,
)

# Start training loop
trainer.fit()
```
