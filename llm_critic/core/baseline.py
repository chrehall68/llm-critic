from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Literal, Union
from llm_critic.data import load_dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
import pickle as pk


def setup_model(model_name: Union[Literal["bert"], Literal["roberta"]]):
    model_map = {"bert": "bert-large-uncased", "roberta": "FacebookAI/roberta-base"}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_map[model_name], num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_map[model_name])
    return model, tokenizer


def run_baseline(
    model_name: Union[Literal["bert"], Literal["roberta"]] = "roberta",
    output_dir: str = "baseline",
):
    ds = load_dataset()
    model, tokenizer = setup_model(model_name)
    splits = ds.train_test_split(test_size=0.2, train_size=0.8)
    train = splits["train"]
    test = splits["test"]  # 20% of original dataset
    splits = train.train_test_split(test_size=0.125, train_size=0.875)
    train = splits["train"]  # 70% of original dataset
    dev = splits["test"]  # 10% of original dataset
    metrics = evaluate.combine(["accuracy", "recall", "precision", "f1"])

    train = train.map(lambda e: tokenizer(e["abstract"], truncation=True)).map(
        lambda e: {"label": 1 if e["accepted"] else 0}
    )
    test = test.map(lambda e: tokenizer(e["abstract"], truncation=True)).map(
        lambda e: {"label": 1 if e["accepted"] else 0}
    )
    dev = dev.map(lambda e: tokenizer(e["abstract"], truncation=True)).map(
        lambda e: {"label": 1 if e["accepted"] else 0}
    )

    args = TrainingArguments(
        run_name=f"{model_name}_baseline",
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        logging_steps=50,
        eval_strategy="epoch",
    )
    trainer = Trainer(
        model,
        args,
        data_collator=DataCollatorWithPadding(tokenizer),
        train_dataset=train,
        eval_dataset=dev,
        # tokenizer=tokenizer,
        compute_metrics=lambda model_preds: metrics.compute(
            predictions=model_preds.predictions.argmax(axis=-1),
            references=model_preds.label_ids,
        ),
    )
    trainer.train()

    outputs = trainer.predict(test)
    metrics = evaluate.combine(["accuracy", "recall", "precision", "f1"])
    results = metrics.compute(outputs.predictions.argmax(axis=-1), test["label"])

    pk.dump(results, open(f"{model_name}_baseline_results.pk", "wb"))
