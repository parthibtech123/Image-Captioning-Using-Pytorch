import torch


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


def print_examples(model, device, dataset):
    """
    Optional sanity check: runs the model on a few hand-picked test images
    (not part of the dataset) so you can eyeball caption quality each epoch.
    Requires a 'test_examples/' folder with a few .jpg images of your choosing.
    Safe to skip early on -- just leave the call commented out in train.py.
    """
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    import os

    test_folder = "test_examples"
    if not os.path.isdir(test_folder):
        model.train()
        return

    for fname in os.listdir(test_folder):
        img = transform(Image.open(os.path.join(test_folder, fname)).convert("RGB")).unsqueeze(0)
        caption = model.caption_image(img.to(device), dataset.vocab)
        print(f"{fname} -> " + " ".join(caption))

    model.train()