from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.gold.gold_config import GOLDConfig
from trl.experimental.gold.gold_trainer import GOLDTrainer


# Model paths
student_model_path = "/workspace/checkpoint/gemma-3-27b-math-rl"
teacher_model_path = "/workspace/checkpoint/gemma-4-31b-it"

# Tokenizers
student_tokenizer = AutoTokenizer.from_pretrained(student_model_path)
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token

# Dataset (ChatML messages 형식)
dataset = load_dataset("/workspace/dataset/vmlu-sdft-v2-preprocessed", split="train")

# Config
training_args = GOLDConfig(
    output_dir="./output/gold-gemma4-to-gemma3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    max_length=2048,
    max_completion_length=1024,
    logging_steps=1,
    save_steps=50,
    bf16=True,
    gradient_checkpointing=True,
    # GKD/GOLD params
    temperature=0.9,
    lmbda=0.0,  # off-policy only (on-policy 없음)
    beta=0.5,
    # ULD + Hybrid
    use_uld_loss=True,
    use_extended_uld=True,
    uld_use_hybrid_loss=True,
    uld_hybrid_matched_weight=1.0,
    uld_hybrid_unmatched_weight=0.0,  # unmatched 제외
    uld_distillation_weight=1.0,
    uld_crossentropy_weight=0.0,
    # Teacher
    teacher_model_name_or_path=teacher_model_path,
    teacher_tokenizer_name_or_path=teacher_model_path,
    teacher_model_init_kwargs={
        "torch_dtype": "bfloat16",
    },
    # Student
    model_init_kwargs={
        "torch_dtype": "bfloat16",
    },
)

# Trainer
trainer = GOLDTrainer(
    model=student_model_path,
    teacher_model=teacher_model_path,
    args=training_args,
    train_dataset=dataset,
    processing_class=student_tokenizer,
)

# Train
trainer.train()
trainer.save_model(training_args.output_dir)
