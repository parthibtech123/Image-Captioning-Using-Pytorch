# Image Captioning using PyTorch

A CNN-to-RNN image captioning model built with PyTorch. An Inception-v3 encoder extracts
image features, which are fed into an LSTM decoder to generate natural language captions.

## Model Architecture

- **Encoder:** Pretrained Inception-v3 (fine-tuned fc layer) → embedding vector
- **Decoder:** LSTM that generates captions word-by-word from the image embedding
- **Training objective:** Cross-entropy loss over the vocabulary, with `<PAD>` tokens ignored

## Project Structure

```
Image-Captioning-Using-Pytorch/
├── flicker8k/              # dataset (NOT included in repo — see setup below)
│   ├── Images/
│   └── captions.txt
├── get_loader.py            # Vocabulary, Dataset, and DataLoader
├── model.py                 # EncoderCNN, DecoderRNN, CNNtoRNN
├── train.py                 # training loop
├── utils.py                 # checkpoint save/load helpers
├── evaluate.py               # CLI evaluation: single image, random samples, BLEU score
├── evaluate_notebook.ipynb   # visual evaluation notebook (image + caption side by side)
└── my_checkpoint.pth.tar     # trained weights (NOT included in repo)
```

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd Image-Captioning-Using-Pytorch
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv flash_env
source flash_env/bin/activate
```

### 3. Install dependencies
```bash
pip install torch torchvision pandas spacy pillow tensorboard nltk matplotlib
python -m spacy download en_core_web_sm
```

### 4. Download the dataset
This project uses the **Flickr8k** dataset. It is not included in this repo due to size.

- Download from Kaggle: https://www.kaggle.com/datasets/adityajn105/flickr8k
- Unzip it and place it in the project root as follows:

```
flicker8k/
├── Images/          # all .jpg files
└── captions.txt     # image,caption pairs
```

## Training

```bash
python train.py
```

- Trains for 100 epochs by default (adjustable in `train.py`)
- Saves a checkpoint (`my_checkpoint.pth.tar`) after every epoch
- Logs training loss to TensorBoard under `runs/flickr`

View training curves:
```bash
tensorboard --logdir runs
```

## Evaluation

### Command line
```bash
# Caption a single image
python evaluate.py --image flicker8k/Images/<some_image>.jpg

# Compare predicted vs actual captions on random samples
python evaluate.py --samples 10

# Compute BLEU-1 through BLEU-4 scores
python evaluate.py --bleu --bleu_n 200
```

### Notebook (visual)
Open `evaluate_notebook.ipynb` in VS Code or Jupyter and run cells top to bottom. It displays
a grid of random images with their predicted and actual captions shown side by side.

## Notes

- Rebuilding the vocabulary from the dataset happens every time `get_loader` or `evaluate.py`
  runs — this takes about a minute since the vocab is derived from all training captions and
  must exactly match what the model was trained with.
- GPU is used automatically if available (`torch.cuda.is_available()`), otherwise falls back to CPU.

## Acknowledgements

Base architecture inspired by the PyTorch image captioning tutorial approach (Inception-v3 encoder + LSTM decoder).