# Text Classification

Build it with TensorFlow and fine-tune BERT.

For the data on this repo, you can get it on: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

You'll be able to use your data, but you'll need to create your code to prepare it. Use my code as a reference.

## Installation

```
git clone https://github.com/RaffelRavionaldo/Text_Classification.git
pip install -r requirements.txt
```

Or you can just clone it and upload the .ipynb file to your drive. Open it with Google Colab.

## Tensorflow.

On this repo, we train the models in 2 ways for the text embeddings.

1. Train it with our own text embeddings.
   
   A. We need to count unique word on our data.
   
   B. Research the longest sentences on how many word on there.
   
   C. From A and B step, we can define this :
   
   ```
    max_words_name = 10000  # Maximum number of words to tokenize for name column
    max_len_name = 200  # Maximum length of sequences for name column
   ```

   for this code, it's on "Just for fun, I do some "analyze" the data" part

  D. Our data will be add a Padding to have "sentences" with max_len_name words.

2. Use GLOVE as our Text Embeddings

  A. Just download it from this website : https://nlp.stanford.edu/projects/glove/ and to load it you can see it on my code.

For this models, I only use 1 layers of bidirectional LSTM and 1 Dense layers with sigmoid activation because the data is simple and we only have 2 classes to predict. if you have more classes. change the activation from sigmoid to softmax.

## BERT

Use it with hugging face, at here we can train all architecture of BERT because it's the parameters we train not to much (only 109.483.778). so i don't use LoRA, PEFT or QLoRA, but if you need it, you can put this code below at your pipeline before train the models and don't forget to install PEFT too : 

### LoRA

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

model_name = "bert-base-uncased"
bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# --- LoRA Configuration ---
lora_config = LoraConfig(
    r=8,           # Rank
    lora_alpha=16, # alpha scala factor
    target_modules=["query", "key", "value"],   # Modules to be adapted/trained (on BERT: attention layers)
                                                # use my code on getting model architecture part to see what modules you can train on this models
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLASSIFICATION",
)

model = prepare_model_for_kbit_training(model) 
bert_model = get_peft_model(model, lora_config)

# Check total parameter will be trained
model.print_trainable_parameters()
# output example: "trainable params: 884,736 || all params: 109,483,520"
```

# QLORA

Same like LoRA, but we add Quantization parts : 

```
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,          # 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,
)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=2
)
```
