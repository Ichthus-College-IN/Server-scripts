#!/usr/bin/python3
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, interleave_datasets
from peft import LoraConfig, get_peft_model
import os

# 1. Geavanceerde configuratie voor Xeon CPU
class XeonRPGConfig:
    def __init__(self):
        self.model_name = "EleutherAI/gpt-j-6B"
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        
        # Meerdere RPG datasets
        self.datasets = [
            ("NeuralNovel/Neural-Story-v1", 0.15),
            ("AtlasUnified/atlas-storyteller", 0.25),
            ("bookcorpus", 0.50),
            ("jaydenccc/AI_Storyteller_Dataset", 0.10)
        ]
        
        self.max_length = 512
        self.num_proc = 16
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            lora_dropout=0.07,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.output_dir = "./xeon_rpg_generator"
        self.num_train_epochs = 5
        self.per_device_batch_size = 2
        self.gradient_accumulation_steps = 4
        self.learning_rate = 3e-5
        self.weight_decay = 0.05

# 2. Geavanceerde data preprocessing
class XeonDataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _process_batch(self, examples):
        processed = []
        for story in examples["text"]:  # Aangepast om met verschillende datasets te werken
            prompt = f"### Verhaal:\n{story}\n### Wat wil je doen?\n1. Ga verder met het verhaal.\n2. Geef je eigen optie.\n"
            processed.append(prompt)
        return {"text": processed}

    def load_data(self):
        datasets = []
        for ds_name, weight in self.config.datasets:
            try:
                print("Loading datasets:")
                ds = load_dataset(ds_name, split="train", num_proc=self.config.num_proc)
                print("Mapping:")
                ds = ds.map(
                    self._process_batch,
                    batched=True,
                    batch_size=1000,
                    num_proc=self.config.num_proc,
                    remove_columns=ds.column_names
                )
                print("Append datasets:")
                datasets.append(ds.with_format("torch"))
            except Exception as e:
                print(f"Fout bij laden {ds_name}: {str(e)}")
        
        return interleave_datasets(
            datasets,
            probabilities=[w for _, w in self.config.datasets],
            stopping_strategy="all_exhausted"
        )

# 3. Geoptimaliseerd model setup
def setup_xeon_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    model = get_peft_model(model, config.lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    
    torch.set_num_threads(config.num_proc)
    os.environ["OMP_NUM_THREADS"] = str(config.num_proc)
    os.environ["MKL_NUM_THREADS"] = str(config.num_proc)
    
    return model

# 4. Parallelle training setup
def train_xeon_model():
    config = XeonRPGConfig()
    processor = XeonDataProcessor(config)
    
    dataset = processor.load_data().train_test_split(test_size=0.05)
    
    model = setup_xeon_model(config)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        save_strategy="epoch",
        logging_steps=50,
        report_to="tensorboard",
        dataloader_num_workers=config.num_proc,
        optim="adafactor",
        group_by_length=True,
        no_cuda=True
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=processor.tokenizer,
        mlm=False,
        pad_to_multiple_of=64
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator
    )
    
    trainer.train()
    model_save_path = "./chatneo_finetuned_cpu"
    os.makedirs(model_save_path, exist_ok=True)  # Create the directory if it doesn't exist

# Save the (fine-tuned) model and tokenizer
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

# 5. Gebruiksvoorbeeld
if __name__ == "__main__":
    train_xeon_model()
