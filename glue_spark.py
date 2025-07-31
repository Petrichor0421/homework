import os
import torch
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, lit, create_map
from pyspark.sql.types import StructType, StructField, ArrayType, LongType, IntegerType, FloatType, StringType
from pyspark.sql.pandas.functions import PandasUDFType
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset # 用于Hugging Face Trainer

# --- 配置 ---
# GLUE 数据集在Ubuntu虚拟机中的路径
GLUE_DATA_PATH = "file:///home/data/glue"

# 微调和推理使用的预训练模型
PRETRAINED_MODEL_NAME = "distilbert-base-uncased"

# 微调后模型保存路径 (在本地文件系统，Spark外)
FINETUNED_MODEL_DIR = "./finetuned_distilbert_models"

# 分词的最大长度
MAX_LENGTH = 128

# Spark 配置
SPARK_APP_NAME = "GLUETextClassification"
# 调整内存配置
SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_MEMORY = "4g"

# 任务分组
TASK_GROUPS = {
    "推理组": ["MNLI", "RTE", "QNLI", "WNLI"],
    "相似性组": ["QQP", "MRPC", "STS-B"],
    "单句组": ["SST-2", "CoLA"]
}

# MNLI 标签映射
MNLI_LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}

# --- 1.初始化 SparkSession ---
spark = SparkSession.builder \
    .appName(SPARK_APP_NAME) \
    .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
    .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

print(f"SparkSession initialized: {spark.version}")

# --- 2.定义通用数据加载函数 ---
def load_glue_task_data(task_name, split="train"):
    """
    根据任务名称和分割类型加载GLUE数据集到Spark DataFrame。
    处理不同任务的列名和标签类型。
    """
    task_path = os.path.join(GLUE_DATA_PATH, task_name)
    file_path = ""
    df = None

    if task_name == "MNLI":
        # MNLI 有 matched 和 mismatched 两个 dev/test 集
        if split == "dev":
            file_path_matched = os.path.join(task_path, "dev_matched.tsv")
            file_path_mismatched = os.path.join(task_path, "dev_mismatched.tsv")
            df_matched = spark.read.csv(file_path_matched, sep='\t', header=True, inferSchema=True)
            df_mismatched = spark.read.csv(file_path_mismatched, sep='\t', header=True, inferSchema=True)
            df = df_matched.union(df_mismatched)
        elif split == "test":  # 通常提交的是 test_matched
            file_path = os.path.join(task_path, "test_matched.tsv")
            df = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)
        else:  # train
            file_path = os.path.join(task_path, f"{split}.tsv")
            df = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)

        # MNLI 的标签是字符串，需要映射
        df = df.select(col("sentence1"), col("sentence2"), col("gold_label").alias("label"), col("index").alias("idx"))
        mapping_expr = create_map([lit(x) for x in sum(MNLI_LABEL_MAP.items(), ())])
        df = df.withColumn("label", mapping_expr[col("label")].cast(IntegerType()))

    elif task_name == "MRPC":
        # MRPC 的原始文件名不同
        if split == "train":
            file_path = os.path.join(task_path, "msr_paraphrase_train.txt")
            df = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)
            df = df.select(col("Quality").alias("label").cast(IntegerType()),
                           col("##1 String").alias("sentence1"),
                           col("#2 String").alias("sentence2"),
                           lit(None).cast(IntegerType()).alias("idx")  # MRPC train无idx，补齐
                           )
        elif split == "dev":
            file_path = os.path.join(task_path, "msr_paraphrase_test.txt")  # MRPC dev用test文件
            df = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)
            df = df.select(col("Quality").alias("label").cast(IntegerType()),
                           col("##1 String").alias("sentence1"),
                           col("#2 String").alias("sentence2"),
                           lit(None).cast(IntegerType()).alias("idx")  # MRPC dev无idx，补齐
                           )
        elif split == "test":  # GLUE测试集通常没有label
            file_path = os.path.join(task_path, "test.tsv")  # 这里假设存在 test.tsv
            df = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)
            df = df.select(col("index").alias("idx"), col("sentence1"), col("sentence2"),
                           lit(None).cast(IntegerType()).alias("label"))


    elif task_name == "CoLA":
        file_path = os.path.join(task_path, f"{split}.tsv")
        # CoLA 的训练集和开发集没有表头，且格式特殊
        if split == "train" or split == "dev":
            df = spark.read.csv(file_path, sep='\t', header=False, inferSchema=True)
            df = df.select(col("_c3").alias("sentence"), col("_c1").alias("label").cast(IntegerType()),
                           lit(None).cast(IntegerType()).alias("idx"))
        else:  # test
            df = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)  # test集有表头
            df = df.select(col("sentence"), lit(None).cast(IntegerType()).alias("label"), col("index").alias("idx"))

    elif task_name == "STS-B":  # 回归任务，标签是浮点数
        file_path = os.path.join(task_path, f"{split}.tsv")
        df = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)
        df = df.select(col("sentence1"), col("sentence2"), col("score").alias("label").cast(FloatType()),
                       col("index").alias("idx"))

    elif task_name == "AX":  # 诊断集，通常是test集
        if split == "test":  # AX通常只有test集，且名为AX_test.tsv
            file_path = os.path.join(task_path, "AX_test.tsv")
            df = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)
            df = df.select(col("sentence1"), col("sentence2"), col("index").alias("idx"),
                           col("gold_label").alias("label"))  # AX的label也是字符串
            mapping_expr = create_map([lit(x) for x in sum(MNLI_LABEL_MAP.items(), ())])  # AX标签与MNLI相同
            df = df.withColumn("label", mapping_expr[col("label")].cast(IntegerType()))
        else:
            print(f"Warning: AX task typically only has a 'test' split (AX_test.tsv). Attempting to load {split}.tsv.")
            file_path = os.path.join(task_path, f"AX_{split}.tsv")  # 尝试加载 AX_train.tsv / AX_dev.tsv
            df = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)
            df = df.select(col("sentence1"), col("sentence2"), col("index").alias("idx"),
                           col("gold_label").alias("label"))
            mapping_expr = create_map([lit(x) for x in sum(MNLI_LABEL_MAP.items(), ())])
            df = df.withColumn("label", mapping_expr[col("label")].cast(IntegerType()))


    else:  # 其他任务如 SST-2, QNLI, QQP, RTE, WNLI
        file_path = os.path.join(task_path, f"{split}.tsv")
        df = spark.read.csv(file_path, sep='\t', header=True, inferSchema=True)
        # 统一字段名，确保有sentence1, sentence2（若无则为null），label, idx
        if "sentence" in df.columns:  # 单句任务
            df = df.select(col("sentence").alias("sentence1"), lit(None).cast(StringType()).alias("sentence2"),
                           col("label").cast(IntegerType()), col("idx").alias("idx"))
        else:  # 句子对任务
            df = df.select(col("sentence1"), col("sentence2"), col("label").cast(IntegerType()),
                           col("idx").alias("idx"))

    if df is None:
        raise ValueError(f"Could not load data for task {task_name} split {split}. Check file path: {file_path}")

    print(f"Loaded {task_name} - {split} data, rows: {df.count()}")
    return df


# --- 3.定义Tokenizer Pandas UDF ---
# 确保 tokenizer 实例只在每个 Executor 上初始化一次
_tokenizer_instance = None
_tokenizer_model_name = None


def get_tokenizer(model_name=PRETRAINED_MODEL_NAME):
    global _tokenizer_instance, _tokenizer_model_name
    if _tokenizer_instance is None or _tokenizer_model_name != model_name:
        print(f"Initializing tokenizer for {model_name} on process {os.getpid()}")
        _tokenizer_instance = AutoTokenizer.from_pretrained(model_name)
        _tokenizer_model_name = model_name
    return _tokenizer_instance


# 定义 UDF 的返回结构
tokenized_schema = StructType([
    StructField("input_ids", ArrayType(LongType())),
    StructField("attention_mask", ArrayType(LongType())),
    StructField("token_type_ids", ArrayType(LongType()))
])


@pandas_udf(tokenized_schema, PandasUDFType.SCALAR)
def tokenize_text_pandas_udf(text_series_1: pd.Series, text_series_2: pd.Series = None) -> pd.DataFrame:
    """
    Pandas UDF 用于在 Spark Executor 上进行文本分词和编码。
    支持单句或句子对输入。
    """
    tokenizer = get_tokenizer()

    texts = text_series_1.tolist()
    text_pairs = text_series_2.tolist() if text_series_2 is not None else None

    # Transformers tokenizer 可以直接处理列表作为批处理输入
    inputs = tokenizer(
        text=texts,
        text_pair=text_pairs,  # 如果 text_pairs 为 None，tokenizer 会自动处理单句
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='np'  # 返回 numpy array 更适合 Pandas UDF
    )

    # 转换为 Python list for Spark ArrayType
    return pd.DataFrame({
        "input_ids": inputs['input_ids'].tolist(),
        "attention_mask": inputs['attention_mask'].tolist(),
        "token_type_ids": inputs['token_type_ids'].tolist() if 'token_type_ids' in inputs else [[]] * len(texts)
    })
