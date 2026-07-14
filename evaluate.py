"""
Evaluation / testing script for the CNN-to-RNN image captioning model.

Usage examples:
    # Caption a single image
    python evaluate.py --image flicker8k/Images/1000268201_693b08cb0e.jpg

    # Caption N random images from the dataset and print predicted vs actual
    python evaluate.py --samples 10

    # Compute BLEU score over a random subset of the dataset
    python evaluate.py --bleu --bleu_n 200
"""

import argparse
import random

import torch
import torchvision.transforms as transforms
from PIL import Image

from get_loader import get_loader
from model import CNNtoRNN

try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def load_model(checkpoint_path, dataset, embed_size=256, hidden_size=256, num_layers=1):
    model = CNNtoRNN(embed_size, hidden_size, len(dataset.vocab), num_layers).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"Loaded checkpoint from '{checkpoint_path}' (trained for {checkpoint.get('step', '?')} steps)")
    return model


def caption_single_image(model, dataset, image_path, max_length=50):
    img = Image.open(image_path).convert("RGB")
    img_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    caption_tokens = model.caption_image(img_tensor, dataset.vocab, max_length=max_length)
    # strip the special tokens for a clean readable sentence
    clean_tokens = [t for t in caption_tokens if t not in ("<SOS>", "<EOS>", "<PAD>")]
    return " ".join(clean_tokens)


def run_single_image(args, dataset, model):
    caption = caption_single_image(model, dataset, args.image)
    print(f"\nImage: {args.image}")
    print(f"Predicted caption: {caption}\n")


def run_random_samples(args, dataset, model):
    """
    Picks N random items straight from the dataset (so we also have
    the ground-truth caption to compare against).
    """
    indices = random.sample(range(len(dataset)), min(args.samples, len(dataset)))

    for i, idx in enumerate(indices, 1):
        img_id = dataset.imgs[idx]
        actual_caption = dataset.captions[idx]
        image_path = f"{dataset.root_dir}/{img_id}"

        predicted = caption_single_image(model, dataset, image_path)

        print(f"[{i}/{len(indices)}] {img_id}")
        print(f"  Actual:    {actual_caption}")
        print(f"  Predicted: {predicted}\n")


def run_bleu(args, dataset, model):
    if not NLTK_AVAILABLE:
        print(
            "nltk is not installed. Install it with:\n"
            "    pip install nltk\n"
            "then rerun with --bleu"
        )
        return

    # group all ground-truth captions by image so each image can have multiple references
    references_by_image = {}
    for img_id, cap in zip(dataset.imgs, dataset.captions):
        references_by_image.setdefault(img_id, []).append(
            dataset.vocab.tokenizer_eng(cap)
        )

    unique_img_ids = list(references_by_image.keys())
    sample_ids = random.sample(unique_img_ids, min(args.bleu_n, len(unique_img_ids)))

    references = []
    hypotheses = []
    smoothing = SmoothingFunction().method4

    print(f"Scoring BLEU on {len(sample_ids)} random images...")
    for i, img_id in enumerate(sample_ids, 1):
        image_path = f"{dataset.root_dir}/{img_id}"
        predicted = caption_single_image(model, dataset, image_path)
        hypotheses.append(predicted.split())
        references.append(references_by_image[img_id])

        if i % 25 == 0 or i == len(sample_ids):
            print(f"  {i}/{len(sample_ids)} done")

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    print("\nBLEU scores:")
    print(f"  BLEU-1: {bleu1:.4f}")
    print(f"  BLEU-2: {bleu2:.4f}")
    print(f"  BLEU-3: {bleu3:.4f}")
    print(f"  BLEU-4: {bleu4:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the image captioning model")
    parser.add_argument("--checkpoint", type=str, default="my_checkpoint.pth.tar")
    parser.add_argument("--root_folder", type=str, default="flicker8k/Images")
    parser.add_argument("--annotation_file", type=str, default="flicker8k/captions.txt")
    parser.add_argument("--image", type=str, help="Path to a single image to caption")
    parser.add_argument("--samples", type=int, default=0, help="Number of random dataset samples to caption")
    parser.add_argument("--bleu", action="store_true", help="Compute BLEU score over a random subset")
    parser.add_argument("--bleu_n", type=int, default=100, help="Number of images to use for BLEU")
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    args = parser.parse_args()

    print("Building vocabulary from dataset (this can take a minute)...")
    _, dataset = get_loader(
        root_folder=args.root_folder,
        annotation_file=args.annotation_file,
        transform=TRANSFORM,
        num_workers=0,
    )

    model = load_model(
        args.checkpoint,
        dataset,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )

    ran_something = False

    if args.image:
        run_single_image(args, dataset, model)
        ran_something = True

    if args.samples > 0:
        run_random_samples(args, dataset, model)
        ran_something = True

    if args.bleu:
        run_bleu(args, dataset, model)
        ran_something = True

    if not ran_something:
        print(
            "Nothing to do. Pass one of:\n"
            "  --image <path>\n"
            "  --samples <N>\n"
            "  --bleu [--bleu_n <N>]"
        )


if __name__ == "__main__":
    main()