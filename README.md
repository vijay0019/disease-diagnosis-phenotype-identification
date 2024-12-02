# NLP - Rare Disease Prediction and Classification
NLP project for predicting and classifying rare diseases given the symptoms and generate precautions.

# Environment
Everything was run using the [UAB Cheaha Supercomputer](https://rc.uab.edu/pun/sys/dashboard/)

# Dataset
We are using the Rare Disease V1 Corpus from [Project NLP4Rare-cm-uc3m](https://github.com/cadovid/nlp4rare). For privacy reasons we couldn't upload the dataset to github, if you need it to run this project, please request it at, [Find the Email Address](https://github.com/isegura/NLP4RARE-CM-UC3M). 

# Model
The model we are using for finetuning is Meta-Llama\Llama 3.2 3B Instruct from [Hugging Face Meta Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).

## Downloading the Model
In the model-download-scripts directory, there are 3 files.

1. `downloader.py` for downloading `meta-llama/Llama-3.2-3B-Instruct`
    - Update the file paths and Hugging Face Tokens
2. `downloader.sh` for passing the job for the download, in the terminal, run `sbatch downloader.sh`
3. `rare_disease_training.sh` for training and finetuning the model with out corpus, in the terminal, run `sbatch rare_disease_training.sh`.

## Training and Fine Tuning Llama 3.2 3B Instruct Model

In the root directory:

1. `rare_disease_training.py` is where all the training and setting of parameters happens
2. To pass the training job, run `sbatch rare_disease_training.sh` found in the model-download-scripts directory.

# Running the Predictor/Classifier

1. Using Jupyter Notebook, run the file `rare_disease_predict.ipynb`
