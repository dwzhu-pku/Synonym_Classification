from datasets import load_dataset
from datasets import load_metric
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
import torch

dataset = load_dataset('csv', data_files={'train': 'Datasets/part01/train.csv',
            'test': 'Datasets/part01/test.csv', 'valid': 'Datasets/part01/valid.csv'})

print(dataset['train'][0])

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

def preprocess_function(examples):
    return tokenizer(examples["text1"], examples["text2"], max_length=512, truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=2)


metric = load_metric('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=7,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

trainer.train()

metrics=trainer.evaluate()
print(metrics)