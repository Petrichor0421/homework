import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

# 导入Colab专用的Google Drive挂载模块
from google.colab import drive

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset

# --- 与主脚本共享的配置 ---
PRETRAINED_MODEL_NAME = "distilbert-base-uncased"

# 在 Colab 环境中，将所有数据保存到 Google Drive
GOOGLE_DRIVE_BASE_DIR = "/content/drive/MyDrive/Colab_GLUE_Models"
FINETUNED_MODEL_DIR = os.path.join(GOOGLE_DRIVE_BASE_DIR, "finetuned_distilbert_models")
HF_CACHE_DIR = os.path.join(GOOGLE_DRIVE_BASE_DIR, "huggingface_cache")
LOGS_DIR = os.path.join(GOOGLE_DRIVE_BASE_DIR, "finetuned_distilbert_logs")

MAX_LENGTH = 128
MNLI_LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}

# 任务分组 - 转换为小写任务名称以匹配 Hugging Face datasets 库的要求
TASK_GROUPS = {
    "推理组": ["mnli", "rte", "qnli", "wnli"],
    "相似性组": ["qqp", "mrpc", "stsb"],
    "单句组": ["sst2", "cola"]
}

# 任务特定的训练参数调整
TASK_SPECIFIC_EPOCHS = {
    "cola": 5, # CoLA 任务增加训练轮次
    "rte": 5   # RTE 任务增加训练轮次
}

# --- 数据加载 ---
def load_and_tokenize_for_training(task_name):
    """
    加载并分词指定GLUE任务的完整数据集。
    """
    if task_name == "mnli":
        # 加载 MNLI 数据集
        train_ds = load_dataset('glue', 'mnli', split='train')
        val_matched_ds = load_dataset('glue', 'mnli', split='validation_matched')
        val_mismatched_ds = load_dataset('glue', 'mnli', split='validation_mismatched')

        dataset = DatasetDict({
            'train': train_ds,
            'validation_matched': val_matched_ds,
            'validation_mismatched': val_mismatched_ds
        })
        num_labels = 3
    elif task_name == "stsb":
        ds = load_dataset('glue', task_name)
        dataset = ds
        num_labels = 1
    else:
        ds = load_dataset('glue', task_name)
        dataset = ds
        if "label" in dataset["train"].features:
            num_labels = dataset["train"].features["label"].num_classes
        else:
            num_labels = None

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    def tokenize_function(examples):
        """
        根据不同的GLUE任务，选择正确的文本列进行分词。
        """
        if "sentence1" in examples and "sentence2" in examples:
            sentences1 = [s if s is not None else "" for s in examples["sentence1"]]
            sentences2 = [s if s is not None else "" for s in examples["sentence2"]]
            return tokenizer(sentences1, sentences2, truncation=True, padding='max_length', max_length=MAX_LENGTH)
        elif "sentence" in examples:
            sentences = [s if s is not None else "" for s in examples["sentence"]]
            return tokenizer(sentences, truncation=True, padding='max_length', max_length=MAX_LENGTH)
        elif "hypothesis" in examples and "premise" in examples:
            sentences1 = [s if s is not None else "" for s in examples["premise"]]
            sentences2 = [s if s is not None else "" for s in examples["hypothesis"]]
            return tokenizer(sentences1, sentences2, truncation=True, padding='max_length', max_length=MAX_LENGTH)
        # QQP 任务处理逻辑：使用 'question1' 和 'question2' 列
        elif "question1" in examples and "question2" in examples:
            sentences1 = [s if s is not None else "" for s in examples["question1"]]
            sentences2 = [s if s is not None else "" for s in examples["question2"]]
            return tokenizer(sentences1, sentences2, truncation=True, padding='max_length', max_length=MAX_LENGTH)
        else:
            missing_cols_info = []
            if "sentence" not in examples: missing_cols_info.append("sentence")
            if not ("sentence1" in examples and "sentence2" in examples): missing_cols_info.append(
                "sentence1/sentence2 pair")
            if not ("hypothesis" in examples and "premise" in examples): missing_cols_info.append(
                "hypothesis/premise pair")
            if not ("question1" in examples and "question2" in examples): missing_cols_info.append(
                "question1/question2 pair")
            raise ValueError(
                f"无法在数据集 {task_name} 中找到文本列。请检查是否存在以下任何列组合: {', '.join(missing_cols_info)}. 实际存在的键: {examples.keys()}")

    tokenized_datasets = DatasetDict()
    for split_name, ds_obj in dataset.items():
        if isinstance(ds_obj, Dataset):
            tokenized_datasets[split_name] = ds_obj.map(tokenize_function, batched=True)
        else:
            tokenized_datasets[split_name] = ds_obj

    if task_name == "mnli":
        def map_mnli_labels(examples):
            """
            将MNLI的标签映射为整数。
            """
            mapped_labels = []
            for i, l in enumerate(examples.get("label", [])):
                # 优先处理整数标签，因为Hugging Face datasets有时会直接提供整数标签
                if isinstance(l, int) and l in [0, 1, 2]:  # MNLI有3个标签，通常是0,1,2
                    mapped_labels.append(l)
                elif isinstance(l, str) and l in MNLI_LABEL_MAP:
                    mapped_labels.append(MNLI_LABEL_MAP[l])
                else:
                    # 如果不是预期的整数或字符串，则视为无效
                    mapped_labels.append(-100)  # 将无效标签设置为 -100
            return {"labels": mapped_labels}

        for split in ['train', 'validation_matched', 'validation_mismatched']:
            if split in tokenized_datasets and "label" in tokenized_datasets[split].features:
                tokenized_datasets[split] = tokenized_datasets[split].map(map_mnli_labels, batched=True,
                                                                          remove_columns=["label"])

            elif split in tokenized_datasets:
                print(f"Warning: {split} split for MNLI does not have a 'label' column. Skipping label mapping.")

    for split_name in tokenized_datasets.keys():
        if "label" in tokenized_datasets[split_name].features:
            tokenized_datasets[split_name] = tokenized_datasets[split_name].rename_column("label", "labels")
        elif task_name != "stsb":
            pass

    final_tokenized_datasets = DatasetDict()
    for split_name, ds in tokenized_datasets.items():
        if "labels" in ds.features:
            ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
            final_tokenized_datasets[split_name] = ds
        else:
            print(f"警告: {task_name} 的 {split_name} 分割不包含 'labels' 列。将不包含 'labels' 进行格式设置。")
            ds.set_format("torch", columns=["input_ids", "attention_mask"])
            final_tokenized_datasets[split_name] = ds

    return final_tokenized_datasets, num_labels


# --- 评估指标函数 ---
def compute_metrics(eval_pred, task_name):
    """
    计算并返回指定任务的评估指标。
    """
    logits, labels = eval_pred
    if task_name == "stsb":
        # 对于STS-B任务，计算皮尔逊和斯皮尔曼相关系数
        preds = logits
        pearson = pearsonr(preds.squeeze(), labels)[0]
        spearman = spearmanr(preds.squeeze(), labels)[0]
        return {"pearson": pearson, "spearman": spearman}

    preds = np.argmax(logits, axis=1)
    # 确保只在 labels 不包含 -100 (被忽略的标签) 时计算指标
    valid_indices = labels != -100
    if np.sum(valid_indices) == 0:
        print(f"警告: 评估集 '{task_name}' 中没有有效标签用于指标计算。")
        # 根据任务类型返回默认值
        if task_name == "cola":
            return {"mcc": 0.0}
        elif task_name in ["mrpc", "qqp"]:
            return {"accuracy": 0.0, "f1": 0.0}
        else:
            return {"accuracy": 0.0}

    labels_filtered = labels[valid_indices]
    preds_filtered = preds[valid_indices]

    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels_filtered, preds_filtered)}
    elif task_name == "mrpc" or task_name == "qqp":
        f1 = f1_score(labels_filtered, preds_filtered, average='weighted')
        acc = accuracy_score(labels_filtered, preds_filtered)
        return {"accuracy": acc, "f1": f1}
    else:  # 其他分类任务，包括MNLI
        return {"accuracy": accuracy_score(labels_filtered, preds_filtered)}


# --- 训练函数 ---
def train_model_for_task(task_name, num_train_epochs=3):
    """
    为指定任务训练模型。
    """
    print(f"\n--- 开始训练任务: {task_name} ---")

    tokenized_datasets, num_labels = load_and_tokenize_for_training(task_name)

    if num_labels is None:
        raise ValueError(f"Task {task_name} does not have a defined number of labels. Cannot initialize model.")

    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels=num_labels)

    # 禁用 W&B 或其他报告工具，确保不会因为网络问题卡住
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["COMET_MODE"] = "DISABLED"

    # 获取任务特定的训练轮次，如果没有则使用 COMMON_EPOCHS
    current_epochs = TASK_SPECIFIC_EPOCHS.get(task_name, num_train_epochs)
    print(f"任务 {task_name} 将训练 {current_epochs} 轮。")

    training_args = TrainingArguments(
        output_dir=os.path.join(FINETUNED_MODEL_DIR, task_name),
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=current_epochs, # 使用任务特定的轮次
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_dir=LOGS_DIR, # 日志目录也指向 Google Drive
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy" if task_name not in ["cola", "stsb"] else (
            "mcc" if task_name == "cola" else "pearson"),
        greater_is_better=True,
        report_to="none",
        no_cuda=False, # 在 Colab 上使用 GPU
        fp16=True, # 在 GPU 上启用混合精度训练，可进一步加速
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation_matched"] if task_name == "mnli" else (
            tokenized_datasets["validation"] if "validation" in tokenized_datasets else None),
        compute_metrics=lambda p: compute_metrics(p, task_name),
    )

    trainer.train()

    final_model_save_path = os.path.join(FINETUNED_MODEL_DIR, task_name, "final_model")
    trainer.save_model(final_model_save_path)
    print(f"模型已保存到: {final_model_save_path}")

    # --- 验证模型文件是否实际存在 ---
    print(f"验证：检查模型文件是否存在于 '{final_model_save_path}'...")
    if os.path.exists(final_model_save_path) and os.path.isdir(final_model_save_path): # 检查是否是目录
        print(f"验证：模型文件确实存在于 '{final_model_save_path}'。")
    else:
        print(f"错误：模型文件未在 '{final_model_save_path}' 找到或不是一个有效目录。")
        print("这表明保存操作可能没有成功，或者保存到了其他位置。")


# --- 执行训练 ---
if __name__ == "__main__":
    # --- 脚本启动诊断与环境配置 ---
    print(f"\n--- 脚本启动诊断与环境配置 ---")

    # 1. 挂载 Google Drive
    print("正在挂载 Google Drive...")
    try:
        drive.mount('/content/drive')
        print("Google Drive 挂载成功。")
    except Exception as e:
        print(f"致命错误：挂载 Google Drive 失败: {e}")
        print("请确保您在 Colab 中授权了 Google Drive 访问权限。脚本将退出。")
        exit()

    print(f"当前脚本运行目录 (os.getcwd()): {os.getcwd()}")

    # 强制设置 Hugging Face 缓存目录到 Google Drive
    os.environ["HF_HOME"] = HF_CACHE_DIR
    print(f"Hugging Face 缓存目录 (HF_HOME) 已设置为: {os.environ['HF_HOME']}")
    print(f"请注意：Hugging Face 会将数据集和预训练模型下载到此目录。")

    # 打印所有关键路径
    print(f"GOOGLE_DRIVE_BASE_DIR (Google Drive 基础目录) 设置为: {GOOGLE_DRIVE_BASE_DIR}")
    print(f"FINETUNED_MODEL_DIR (模型保存目录) 设置为: {FINETUNED_MODEL_DIR}")
    print(f"LOGS_DIR (日志保存目录) 设置为: {LOGS_DIR}")

    # 确保所有目标目录存在
    directories_to_create = [GOOGLE_DRIVE_BASE_DIR, FINETUNED_MODEL_DIR, HF_CACHE_DIR, LOGS_DIR]
    for d in directories_to_create:
        print(f"尝试创建目录: {d}")
        try:
            os.makedirs(d, exist_ok=True)
            print(f"目录 '{d}' 创建成功或已存在。")
        except Exception as e:
            print(f"致命错误：创建目录 '{d}' 时出错: {e}")
            print("请检查 Google Drive 是否有足够的空间，以及路径是否正确。脚本将退出。")
            exit() # 如果任何关键目录创建失败，立即退出

    # 在 FINETUNED_MODEL_DIR 根目录创建一个测试文件，以验证基本写入权限和盘符可访问性
    test_file_path = os.path.join(FINETUNED_MODEL_DIR, "_python_test_write_gdrive.txt")
    print(f"尝试在模型保存主目录 '{FINETUNED_MODEL_DIR}' 中创建测试文件: {test_file_path}")
    try:
        with open(test_file_path, "w") as f:
            f.write("This is a test file created by Python script to verify Google Drive write access.")
        print(f"测试文件 '{test_file_path}' 创建成功。")
        os.remove(test_file_path)
        print(f"测试文件 '{test_file_path}' 删除成功。")
    except Exception as e:
        print(f"致命错误：在 '{FINETUNED_MODEL_DIR}' 中创建/删除测试文件时出错: {e}")
        print("这可能表明 Google Drive 挂载有问题，或者您没有足够的写入权限。脚本将退出。")
        exit()

    print(f"--- 脚本启动诊断与环境配置完成 ---\n")
    # --- 深度调试代码结束 ---

    COMMON_EPOCHS = 3 # 默认训练轮次

    for group_name, tasks in TASK_GROUPS.items():
        print(f"\n--- 训练任务组: {group_name} ---")
        for task in tasks:
            if task == "wnli":
                print(f"Skipping WNLI due to known issues and small dataset size.")
                continue

            try:
                train_model_for_task(task, num_train_epochs=COMMON_EPOCHS)
            except Exception as e:
                print(f"训练 {task} 时出错: {e}")
                print(f"错误详情: {e}")
                print("请确认您的 GLUE 本地数据集文件结构和格式与 Hugging Face datasets 库预期的一致。")

    print("\n所有训练任务完成！")
