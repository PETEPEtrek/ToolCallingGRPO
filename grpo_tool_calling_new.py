import os
import re
import json
import torch
import pandas as pd
from datasets import Dataset
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import setup_chat_format, GRPOConfig, GRPOTrainer
from rouge_score import rouge_scorer


# 1) ПАРАМЕТРЫ МОДЕЛИ И LoRA
# ───────────────────────────────────────────────────────────────
MODEL_PATH = "models/petepetrek/Vikhrmodels/QVikhr-3-4B-Instruction/"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32

# ───────────────────────────────────────────────────────────────
# 2) TOOL NAMES AND REGEX PATTERNS
# ───────────────────────────────────────────────────────────────
VALID_TOOLS = [
    "core_memory_add_human",
    "core_memory_edit_human", 
    "core_memory_add_persona",
    "core_memory_edit_persona",
    "knowledge_add_fact",
    "knowledge_edit_fact"
]

# Regex для поиска tool calls
TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*\{.*?"name":\s*"([^"]+)".*?"arguments":\s*\{.*?"(?:information|content)":\s*"([^"]*)".*?\}\s*\}\s*</tool_call>',
    re.DOTALL | re.IGNORECASE
)

# Глобальные переменные для отладки
PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5

# ───────────────────────────────────────────────────────────────
# 3) REWARD FUNCTIONS
# ───────────────────────────────────────────────────────────────

def check_tool_call_format(completions, **kwargs):
    """Проверяет правильность формата tool calls"""
    scores = []
    
    for completion in completions:
        response = completion[0]["content"]
        score = 0.0
        
        # Ищем все tool calls в ответе
        tool_calls = TOOL_CALL_PATTERN.findall(response)
        
        for tool_name, info in tool_calls:
            # Проверяем, что имя инструмента правильное
            if tool_name in VALID_TOOLS:
                score += 1.0
            else:
                score -= 0.5
                
            # Проверяем, что информация не пустая
            if info.strip():
                score += 0.5
            else:
                score -= 0.5
        
        scores.append(score)
    
    return scores

def check_tool_call_count(completions, **kwargs):
    """Проверяет правильное количество tool calls и дает бонус 0.5 за каждый правильный"""
    scores = []
    
    # Получаем ground truth из kwargs 
    batch_data = kwargs.get('tool_calls', [])
    
    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        score = 0.0
        
        # Подсчитываем количество найденных tool calls
        found_tool_calls = TOOL_CALL_PATTERN.findall(response)
        
        # Получаем ground truth для этого примера
        if i < len(batch_data):# and 'tool_calls' in batch_data[i]:
            gt_tool_calls = batch_data[i]
            print("CHECK COUNT TOOL CALLS: ", gt_tool_calls)
            expected_count = gt_tool_calls.count('<tool_call>')
            found_count = len(found_tool_calls)
            print(f"Expected {expected_count}, Found {found_count}")
            
            # Даем бонус за правильное количество
            if found_count == expected_count:
                score += 0.5 * expected_count
            else:
                # Штрафуем за неправильное количество
                score -= abs(found_count - expected_count) * 0.25
        
        scores.append(score)
    
    return scores

def check_information_quality_rouge(completions, **kwargs):
    """Проверяет качество информации через ROUGE метрику"""
    print("ROUGE")
    scores = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Получаем ground truth из kwargs 
    #print("KWARGS KEYS:", list(kwargs.keys()))
    batch_data = kwargs.get('tool_calls', [])
    print("="*80)
    print(batch_data)
    print("="*80) 
    for i, completion in enumerate(completions):
        response = completion[0]["content"]
        score = 0.0
        
        # Извлекаем информацию из сгенерированного ответа
        found_tool_calls = TOOL_CALL_PATTERN.findall(response)
        print('TOOL CALLS: ', found_tool_calls) 
        # Получаем ground truth для этого примера
        if i < len(batch_data): # and 'tool_calls' in batch_data[i]:
            #gt_tool_calls = batch_data[i]['tool_calls']
            gt_tool_calls = batch_data[i]
            print("RAW TOOL CALLS: ", gt_tool_calls)
            gt_tool_calls_parsed = TOOL_CALL_PATTERN.findall(gt_tool_calls)
            print('PARSED TOOL CALLS: ', gt_tool_calls_parsed)
            # Сравниваем каждую найденную информацию с ground truth
            for found_tool, found_info in found_tool_calls:
                best_rouge_score = 0.0
                
                # Ищем лучшее совпадение по инструменту и информации
                for gt_tool, gt_info in gt_tool_calls_parsed:
                    if found_tool == gt_tool:
                        # Вычисляем ROUGE score
                        rouge_scores = scorer.score(gt_info, found_info)
                        rouge_l_score = rouge_scores['rougeL'].fmeasure
                        best_rouge_score = max(best_rouge_score, rouge_l_score)
                print("BEST ROUGE SCORE: ", best_rouge_score)
                score += best_rouge_score * 2.0  # Умножаем на 2 для большего веса
        
        scores.append(score)
    
    return scores

def debug_print_examples(completions, **kwargs):
    """Функция для отладки - печатает примеры"""
    global PRINTED_TIMES, PRINT_EVERY_STEPS
    
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        batch_data = kwargs.get('batch', [])
        print("=" * 80)
        print("DEBUGGING EXAMPLE:")
        print("=" * 80)
        print("Generated response:")
        print(completions[0][0]["content"])
        print("\n" + "-" * 40)
        if batch_data and 'tool_calls' in batch_data[0]:
            print("Ground truth tool calls:")
            print(batch_data[0]['tool_calls'])
        print("=" * 80)
    
    PRINTED_TIMES += 1
    
    # Возвращаем нулевые scores, так как это только для отладки
    return [0.0] * len(completions)

# ───────────────────────────────────────────────────────────────
# 4) SYSTEM PROMPT
# ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are given a text that contains information about a user, information about your persona, and general knowledge. Some information may be modified or changed. You need to track and extract this information using the appropriate tools.

Extract information and use the following tools:

**For user information (facts about the user):**
- Use `core_memory_add_human` for new/original user information
- Use `core_memory_edit_human` for modified/changed user information

**For your persona information (facts about yourself):**
- Use `core_memory_add_persona` for new/original persona information  
- Use `core_memory_edit_persona` for modified/changed persona information

**For general knowledge/facts:**
- Use `knowledge_add_fact` for new/original facts
- Use `knowledge_edit_fact` for modified/changed facts

**Examples:**

User fact (original):
<tool_call>
{"name": "core_memory_add_human", "arguments": {"information": "i love playing guitar"}}
</tool_call>

User fact (modified):
<tool_call>
{"name": "core_memory_edit_human", "arguments": {"information": "i don't love playing guitar"}}
</tool_call>

Persona fact (original):
<tool_call>
{"name": "core_memory_add_persona", "arguments": {"information": "you enjoy hiking on weekends"}}
</tool_call>

Persona fact (modified):
<tool_call>
{"name": "core_memory_edit_persona", "arguments": {"information": "you don't enjoy hiking on weekends"}}
</tool_call>

General knowledge (original):
<tool_call>
{"name": "knowledge_add_fact", "arguments": {"information": "cats are mammals"}}
</tool_call>

General knowledge (modified):
<tool_call>
{"name": "knowledge_edit_fact", "arguments": {"content": "cats are not mammals"}}
</tool_call>

Extract all relevant information from the given text and use the appropriate tools."""

# ───────────────────────────────────────────────────────────────
# 5) MAIN FUNCTION
# ───────────────────────────────────────────────────────────────

def main():
    # ───────────────────────────────
    # 5.1) Load dataset
    # ───────────────────────────────
    print("Loading dataset...")
    df = pd.read_csv("datasets/training_dataset_1000_enhanced.csv")
    print(f"Dataset loaded: {len(df)} rows")
    
    # ───────────────────────────────
    # 5.2) Prepare dataset for training
    # ───────────────────────────────
    def prepare_dataset_row(row):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["sentences"]},
            ],
            "tool_calls": row["tool_calls"]
        }
    
    # Convert to HuggingFace Dataset
    dataset_dict = []
    for _, row in df.iterrows():
        dataset_dict.append(prepare_dataset_row(row))
    
    dataset = Dataset.from_list(dataset_dict)
    print(f"Dataset prepared: {len(dataset)} examples")
    
    # ───────────────────────────────
    # 5.3) Load model and tokenizer
    # ───────────────────────────────
    print("Loading model and tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # ───────────────────────────────
    # 5.4) Apply LoRA
    # ───────────────────────────────
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)
    
    # ───────────────────────────────
    # 5.5) Calculate max prompt length
    # ───────────────────────────────
    print("Calculating max prompt length...")
    max_prompt_length = 0
    for example in dataset:
        tokens = tokenizer.apply_chat_template(
            example["prompt"], 
            add_generation_prompt=True, 
            tokenize=True
        )
        max_prompt_length = max(max_prompt_length, len(tokens))
    
    print(f"Max prompt length: {max_prompt_length}")
    
    # ───────────────────────────────
    # 5.6) Split dataset
    # ───────────────────────────────
    dataset = dataset.shuffle(seed=42)
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")
    
    # ───────────────────────────────
    # 5.7) Training configuration
    # ───────────────────────────────
    training_args = GRPOConfig(
        learning_rate=5e-6,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        logging_steps=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_prompt_length=max_prompt_length + 10,
        max_completion_length=MAX_SEQ_LENGTH - max_prompt_length - 10,
        max_steps=1000,
        save_steps=200,
        max_grad_norm=0.1,
        report_to="tensorboard",
        logging_dir="runs_tool_calling3",
        output_dir="outputs_tool_calling3",
    )
    
    # ───────────────────────────────
    # 5.8) Create trainer
    # ───────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            check_tool_call_format,
            check_tool_call_count,
            check_information_quality_rouge,
            #debug_print_examples,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # ───────────────────────────────
    # 5.9) Start training
    # ───────────────────────────────
    print("Starting training...")
    trainer.train()
    
    # ───────────────────────────────
    # 5.10) Save model
    # ───────────────────────────────
    print("Saving model...")
    trainer.save_model("outputs_tool_calling3/final_model")
    
    print("Training completed!")

if __name__ == "__main__":
    main() 
