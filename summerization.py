from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from transformers import create_optimizer, AdamWeightDecay
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback
from transformers.keras_callbacks import PushToHubCallback
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM




billsum = load_dataset("billsum", split="ca_test")

billsum["train"][0]


checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = "summarize: "


def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_billsum = billsum.map(preprocess_function, batched=True)


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")



rouge = evaluate.load("rouge")




def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}



optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)

tf_train_set = model.prepare_tf_dataset(
    tokenized_billsum["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    tokenized_billsum["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)



model.compile(optimizer=optimizer)  # No loss argument!


metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)

push_to_hub_callback = PushToHubCallback(
    output_dir="my_awesome_billsum_model",
    tokenizer=tokenizer,
)

callbacks = [metric_callback, push_to_hub_callback]
model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
summarizer(text)

tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_billsum_model")
inputs = tokenizer(text, return_tensors="tf").input_ids


model = TFAutoModelForSeq2SeqLM.from_pretrained("stevhliu/my_awesome_billsum_model")
outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)
tokenizer.decode(outputs[0], skip_special_tokens=True)