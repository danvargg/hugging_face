"""Hugging face transfer learning."""
import transformers
from datasets import load_dataset, Dataset, DatasetDict
from transformers import DistilBertTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import numpy as np

transformers.logging.set_verbosity_error()

# Use pretrained model checkpoint from Huggingface
model_name = "distilbert-base-uncased"

# Use pre-labeled dataset from huggingface
dataset_name = "poem_sentiment"

poem_sentiments = load_dataset(dataset_name)

# Apache Arrow format
print(poem_sentiments)
print(poem_sentiments["test"][20:25])

print("\nSentiment Labels used", poem_sentiments["train"].features.get("label").names)

# Encoding text
db_tokenizer = DistilBertTokenizer.from_pretrained(model_name)


def tokenize(batch):
    return db_tokenizer(batch["verse_text"], padding=True, truncation=True)


enc_poem_sentiment = poem_sentiments.map(tokenize, batched=True, batch_size=None)

print(enc_poem_sentiment["train"][0:5])

# Explore input IDs and Attention Mask
print("Text :", enc_poem_sentiment["train"][1].get("verse_text"))
print("\nInput Map :", enc_poem_sentiment["train"][1].get("input_ids"))
print("\nAttention Mask :", enc_poem_sentiment["train"][1].get("attention_mask"))

print("\nTotal tokens: ", len(enc_poem_sentiment["train"][1].get("input_ids")))
print("Non Zero tokens: ", len(list(filter(lambda x: x > 0, enc_poem_sentiment["train"][1].get("input_ids")))))
print("Attention = 1: ", len(list(filter(lambda x: x > 0, enc_poem_sentiment["train"][1].get("attention_mask")))))

# Separate training and validation sets
training_dataset = enc_poem_sentiment["train"]
validation_dataset = enc_poem_sentiment["validation"]

print("\nColumn Names : ", training_dataset.column_names)
print("\nFeatures : ", training_dataset.features)

labels = training_dataset.features.get("label")
num_labels = len(labels.names)

# Load transformer checkpoint from huggingface
sentiment_model = (TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels))

# sentiment_model.get_config()

# Freeze the first layer if needed
sentiment_model.layers[0].trainable = True

# Add/remove layers if needed.
# sentiment_model.layers [append()/insert()/remove()]

print(sentiment_model.summary())

batch_size = 64
tokenizer_columns = db_tokenizer.model_input_names

# Convert to TF tensors
train_dataset = training_dataset.to_tf_dataset(
    columns=tokenizer_columns, label_cols=["label"], shuffle=True,
    batch_size=batch_size)
val_dataset = validation_dataset.to_tf_dataset(
    columns=tokenizer_columns, label_cols=["label"], shuffle=False,
    batch_size=batch_size)

sentiment_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy())

sentiment_model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=5)

# Input data for interference to predict sentiment
# the "label" array is not needed for inference, but added to provide true labels for comparison
infer_data = {
    'id': [0, 1],
    'verse_text': [
        'and be glad in the summer morning when the kindred ride on their way',
        'how hearts were answering to his own'
    ],
    'label': [1, 0]
}

infer_dataset = Dataset.from_dict(infer_data)

ds_dict = DatasetDict()
ds_dict["infer"] = infer_dataset

print(ds_dict)

# Encode the dataset, similar to training
enc_dataset = ds_dict.map(tokenize, batched=True, batch_size=None)

# Convert to Tensors
infer_final_dataset = enc_dataset["infer"].to_tf_dataset(
    columns=tokenizer_columns, shuffle=True, batch_size=batch_size
)

print(infer_final_dataset)

# Predict with the model
predictions = sentiment_model.predict(infer_final_dataset)

print(predictions.logits)

pred_label_ids = np.argmax(predictions.logits, axis=1)

for i in range(len(pred_label_ids)):
    print("Poem =", infer_data["verse_text"][i],
          " Predicted=", labels.names[pred_label_ids[i]],
          " True-Label=", labels.names[infer_data["label"][i]])
