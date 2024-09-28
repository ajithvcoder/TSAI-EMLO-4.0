import json
import time
import random
import torch
import argparse
from torchvision import datasets, transforms
from pathlib import Path
from PIL import Image
from model.model import Net
import os


def infer(model, dataset, save_dir, num_samples=5):
    model.eval()
    results_dir = Path("/opt/mount/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), num_samples)
    for idx in indices:
        image, _ = dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1, keepdim=True).item()

        img = Image.fromarray(image.squeeze().numpy() * 255).convert("L")
        img.save(results_dir / f"{idx}_{pred}.png")


def main():
    parser = argparse.ArgumentParser(description="MNIST Evaluation Script")
    save_dir = "/opt/mount"
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save_model", default="model", help="checkpoint will be saved in this directory"
    )
    
    args = parser.parse_args()
    # init model and load checkpoint here
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    model = Net().to(device)
    model.load_state_dict(torch.load(os.path.join("/opt/mount",args.save_model,"mnist_cnn.pt")))

	# create transforms and test dataset for mnist
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # create MNIST test dataset and loader
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

    infer(model, dataset1, save_dir)
    print("Inference completed. Results saved in the 'results' folder.")


if __name__ == "__main__":
    main()
