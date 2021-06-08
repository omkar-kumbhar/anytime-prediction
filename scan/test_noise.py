import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sresnet
import torch.nn.functional as F
import numpy as np

import json
import time
import os

from custom_transforms import AddGaussianNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(100)
torch.cuda.manual_seed(100)

parser = argparse.ArgumentParser(description='SCAN Noise Inference')
parser.add_argument('--tag', default="noise-contrast-gray", type=str, help="Experiment ID")
parser.add_argument('--data-root', type=str, help="Path to dataset root or where it should be downloaded")
parser.add_argument('--load-path', type=str, help='path to load saved model from')
parser.add_argument('--depth', default=18, type=int, help="Type of ResNet backbone. Options: 18 (default), 9, 34")
parser.add_argument('--class_num', default=10, type=int, help="Number of classification categories (default: 10)")
parser.add_argument('--epoch', default=200, type=int, help="number of training epochs (default: 200)")
parser.add_argument('--lambda_KD', default=0.5, type=float, help="self-distillation coefficient in loss function (default: 0.5)")
parser.add_argument('--noise', default=0.0, type=float, help="Gaussian SD value to apply")
args = parser.parse_args()
print(args)


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs/3.0, dim=1)
    softmax_targets = F.softmax(targets/3.0, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def judge(tensor, c, dic):
    maxium = torch.max(tensor)
    if float(maxium) > dic[c]:
        return True
    else:

        return False


BATCH_SIZE = 256
LR = 0.1

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    AddGaussianNoise(0.0, args.noise),
])

trainset, testset = None, None
if args.class_num == 100:
    print("dataset: CIFAR100")
    testset = torchvision.datasets.CIFAR100(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform_test
    )
if args.class_num == 10:
    print("dataset: CIFAR10")
    testset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform_test
    )
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

net = None
if args.depth == 9:
    net = sresnet.resnet9(num_classes=args.class_num, align="CONV", pretrained=False)
    net_ops = [76758016., 80550912., 96201728., 103437312., 173136896.]
    print("using resnet 9")
if args.depth == 18:
    net = sresnet.resnet18(num_classes=args.class_num, align="CONV", pretrained=False)
    net_ops = [190856192., 308321280., 437431296., 558019584., 627719168.]
    print("using resnet 18")
if args.depth == 34:
    net = sresnet.resnet34(num_classes=args.class_num, align="CONV", pretrained=False)
    net_ops = [266943488., 535993344., 967683072., 1163842560., 1233542144.]
    print("using resnet 34")

net.to(device)
net.load_state_dict(torch.load(args.load_path))

if __name__ == "__main__":
    best_acc = 0

    # create experiment directory
    expt_id = "{}_{}_resnet{}_cifar{}_noise{}".format(
        args.tag,
        time.strftime('%Y.%m.%d_%H.%M.%S'),
        args.depth,
        args.class_num,
        args.noise
    )
    root = os.path.join('expt/test', expt_id)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), 'w') as f:
        json.dump(vars(args), f)

    print("Waiting Test!")
    for c in range(0,5):
        caught = [0, 0, 0, 0, 0]
        with torch.no_grad():
            correct4, correct3, correct2, correct1, correct0 = 0, 0, 0, 0, 0
            predicted4, predicted3, predicted2, predicted1, predicted0 = 0, 0, 0, 0, 0
            correct = 0.0
            total = 0.0
            right = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs, feature_loss = net(images)
                ensemble = sum(outputs) / len(outputs)
                outputs.reverse()

                for index in range(len(outputs)):
                    outputs[index] = F.softmax(outputs[index])

                for index in range(images.size(0)):
                    ok = False

                    if c < 4:
                        logits = outputs[c][index]
                        caught[c] += 1
                        predict = torch.argmax(logits)
                        if predict.cpu().numpy().item() == labels[index]:
                            right += 1

                        ok = True

                    if not ok: # i.e. c == 5
                        caught[-1] += 1
                        #   print(index, "ensemble")
                        logits = ensemble[index]
                        predict = torch.argmax(logits)
                        if predict.cpu().numpy().item() == labels[index]:
                            right += 1

                total += float(labels.size(0))

            # FLOPs (only for resnet18)

            print("caughts:", caught)
            acceleration_ratio = 1/((0.32 * caught[0] + 0.53* caught[1] + 0.76*caught[2] + 1.0 * caught[3] + 1.07 * caught[4])/total)
            accr_msg = "Acceleration ratio: %.4f " % acceleration_ratio

            ops = sum([i*j for i,j in zip(net_ops, caught)])/total
            ops_msg = 'FLOPs: %.4f ' % ops

            print(ops_msg)
            print(accr_msg)

            tst_msg = 'Test Set Accuracy:  %.4f%% ' % (100 * right / total) 
            print(tst_msg)
            print("-----------------")

            with open(os.path.join(root, "summary.log"), 'a') as f:
                f.write(tst_msg + accr_msg + ops_msg + '\n')
