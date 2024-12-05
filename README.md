# NLP - Final Project

NLP project: Identifying the phenotype terms using Bio_ClinicalBERT and then performing diagnosis given the identified phenotypes using meta-llama/Llama-3.2-3B-Instruct.

# Environment

Everything was run using the [UAB Cheaha Supercomputer](https://rc.uab.edu/pun/sys/dashboard/)

# Model

The model we are using for finetuning is Meta-Llama\Llama 3.2 3B Instruct from [Hugging Face Meta Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and Bio_ClinicalBert

# Llama 3.2 3B Rare Disease Prediction and Classification

## Dataset

We are using the Rare Disease V1 Corpus from [Project NLP4Rare-cm-uc3m](https://github.com/cadovid/nlp4rare). For copyright reasons we couldn't upload the dataset to github, if you need it to run this project, please request it at, [Find the Email Address](https://github.com/isegura/NLP4RARE-CM-UC3M).

## Downloading the Model

In the model-download-scripts directory, there are 3 files.

1. To download Meta-Llama\Llama_3.2_3B_Instruct, edit the file `model-download-scripts\download.sh` to point to `model-download-scripts\download.py` under `python` keyword.
2. Add the Hugging Face Token to `hf_token` in `model-download-scripts\download.py`
3. Change the `save_dir` in `model-download-scripts\download.py`
4. Store the `model-download-scripts\download.sh` in the file under `/home/<user>` directory
5. In cheaha cluster terminal, run `sbatch download.sh` in the `/home/<user>` directory

## Training and Fine Tuning Llama 3.2 3B Instruct Model

In the root directory:

1. `rare_disease_training.py` is where all the training and setting of parameters happens
   - Update the `base-model` and `new-model` paths.
   - Update the `train_texts, train_annotations`, `ev_texts, dev_annotations`, and `test_texts, test_annotations`
2. To pass the training job,
   - Store the `model-download-scripts\rare_disease_training.sh` in the `/home/<user>` directory
   - Edit the file `model-download-scripts\are_disease_training.sh` to point to `rare_disease_training.py` under `python` keyword
   - In cheaha cluster terminal, run `sbatch rare_disease_training.sh` in the `/home/<user>` directory

## Testing the Predictor/Classifier

1. Using Jupyter Notebook, set the `partition` to `pascalnodes`.
2. Run the file `rare_disease_predict.ipynb`

# Bio_ClinicalBERT

## Models

1. `model_scripts/diagnosisANDphonotype/bert_finetune_raredis.ipynb`
2. `model_scripts/diagnosisANDphonotype/bert_finetune_synthetic.ipynb`

NB: I have included the download and training output files for results under `llama_output` directory.
