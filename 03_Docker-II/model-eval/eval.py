import json
import torch
import torch.nn.functional as F
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from model.model import Net
import os
import time


def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item() # sum up batch loss
            pred = output.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    eval_results = {"Average loss": test_loss, "Accuracy": 100. * correct / len(data_loader.dataset)}
    return eval_results

def test(args, model, device, dataset, dataloader_kwargs):
    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    return test_epoch(model, device, test_loader)

def main():
    parser = argparse.ArgumentParser(description="MNIST Evaluation Script")
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--save_model", default="./", help="checkpoint will be saved in this directory"
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    kwargs = {
        "batch_size": args.test_batch_size,
        "num_workers": 1,
        "shuffle": True,
    }
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

    # create model and load state dict
    # create model and setup mp
    torch.manual_seed(args.seed)
    model = Net().to(device)
    model.load_state_dict(torch.load(os.path.join("/opt/mount",args.save_model,"mnist_cnn.pt")))

    # test epoch function call
    eval_results = test(args, model, device, dataset2, kwargs)

    with (Path(os.path.join("/opt/mount", args.save_model, "eval_results.json"))).open("w") as f:
        json.dump(eval_results, f)


if __name__ == "__main__":
    main()
