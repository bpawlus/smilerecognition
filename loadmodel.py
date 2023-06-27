import argparse
from model import DeepSmileNet
import torch
from constants import dirsdict
from constants import valuesdict
import os

parser = argparse.ArgumentParser(description='RealSmileNet training')


def print_module_items(name, model_items):
    print(f"  {name}:")
    for module_pos, module in model_items:
        print(f"    P. {module_pos}: {module}")
    print()

def main():
    args = parser.parse_args()

    f = ["".join(feature) for feature in args.f]
    model_state_dir = args.model.replace("\\", "/")
    model = DeepSmileNet(f)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Model state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print("\n")

    print("Optimizer state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
    print("\n")

    checkpoint = torch.load(model_state_dir)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    print("Model parameters:")
    for param in model.parameters():
        print(f"{param.data.size()}: {param.data}")
    print("\n")

    print("Model items:")
    if "videos" in f:
        print_module_items("TSA", model.TSA._modules.items())
        print_module_items("FPN", model.FPN._modules.items())
        print_module_items("ConvLSTMLayer", model.ConvLSTMLayer._modules.items())
        print_module_items("Classification", model.Classification._modules.items())

    if "aus" in f:
        print_module_items("AUsLSTM", model.AUsLSTM._modules.items())
        print_module_items("ClassificationAUs", model.ClassificationAUs._modules.items())

    if "si" in f:
        print_module_items("SILSTM", model.SILSTM._modules.items())
        print_module_items("ClassificationSI", model.ClassificationSI._modules.items())

    print_module_items("ClassificationCat", model.ClassificationCat._modules.items())
    print("\n")

main()