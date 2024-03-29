#!/usr/bin/env python3

"""once.py
This script can be used to once run Deep Fingerprinting (DF) on the goodenough
dataset and produce some metrics. Right now only working on extracted cells, but
made to be straightforward to extend.
There are many parameters to tweak, see the arguments and help below.  Works
well on my machine with a RTX 2070 (see batch size based on memory). 
Supports saving and loading datasets as well as models.
The DF implementation is ported to PyTorch with inspiration from
https://github.com/lin-zju/deep-fp .
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import datetime
import pickle
import csv
import shared

ap = argparse.ArgumentParser()
# load and save dataset/model
ap.add_argument("--ld", required=False, default="",
    help="load dataset from pickle, provide path to pickled file")
ap.add_argument("--sd", required=False, default="",
    help="save dataset, provide path to dump pickled file")
ap.add_argument("--lm", required=False, default="",
    help="load model from pickle, provide path to pickled file")
ap.add_argument("--sm", required=False, default="",
    help="save model, provide path to dump pickled file")

## extra output
ap.add_argument("--csv", required=False, default=None,
    help="save resulting metrics in provided path in csv format")
ap.add_argument("--extra", required=False, default="",
    help="value of extra column in csv output")

# extract/train new dataset/model
ap.add_argument("--ed", required=False, default="",
    help="extract dataset, path with {monitored,unmonitored} subfolders")
ap.add_argument("--train", required=False, default=False,
    action="store_true", help="train model")

# experiment parameters
ap.add_argument("--epochs", required=False, type=int, default=30,
    help="the number of epochs for training")
ap.add_argument("--batchsize", required=False, type=int, default=750,
    help="batch size")
ap.add_argument("-f", required=False, type=int, default=0,
    help="the fold number (partition offset)")
ap.add_argument("-l", required=False, type=int, default=5000,
    help="max input length used in DF")
ap.add_argument("-z", required=False, default="",
    help="zero each sample between sample[zero], e.g., 0:10 for the first 10 cells")

ap.add_argument("--dt", action='store_true',
    help="run Directional Time instead of just Direction")
ap.add_argument("--gaps", action='store_true',
    help="run with gaps")
ap.add_argument("--size", required=False, type=int, default=1,
    help="size of dataset")
ap.add_argument("--st", required=True, type=int, default=500,
    help="amount of sub-traces, DS-TS=500, DS-WA=450, DS-DF=5000")

# dataset dimensions
ap.add_argument("-c", required=False, type=int, default=100,
    help="the number of monitored classes")
ap.add_argument("-p", required=False, type=int, default=10,
    help="the number of partitions")
ap.add_argument("-s", required=False, type=int, default=20,
    help="the number of samples")
args = vars(ap.parse_args())

def now():
    return datetime.datetime.now().strftime("%H:%M:%S")

def main():
    if (
        (args["ld"] == "" and args["ed"] == "") or 
        (args["ld"] != "" and args["ed"] != "")
    ):
        sys.exit(f"needs exactly one of --ld and --ed")

    dataset, labels = {}, {}
    if args["ld"] != "":
        print(f"attempting to load dataset from pickle file {args['ld']}")
        dataset, labels = pickle.load(open(args["ld"], "rb"))
        # flatten dataset with extra details (generated by tweak.py)
        for k in dataset:
            dataset[k][0][dataset[k][0] > 1.0] = 1.0
            dataset[k][0][dataset[k][0] < -1.0] = -1.0

    else:
        if args["train"] and not os.path.isdir(args["ed"]):
            sys.exit(f"{args['ed']} is not a directory")
        if args["size"]  > len(os.listdir(args["ed"])):
            sys.exit(f"size is too large ({args['size']})")
            
        
        print(f"{now()} starting to load dataset from {args['ed']}...")
        
        if not args["train"]: # Testing Directional Time with the non-sim BWR5
            dataset, labels, sample_tracker = shared.load_dataset_BWR5(
                args["ed"],
                args["l"],
                args["dt"],
                args["gaps"],
                args["size"],
                shared.extract_BWR5 # shared.extract_BWR5_DT
            )
        else:
            dataset, labels, sample_tracker = shared.load_dataset_BWR5(
                args["ed"],
                args["l"],
                args["dt"],
                args["gaps"],
                args["size"],
                shared.extract_BWR5
            )
        print(f"{now()} loaded dataset.")
            
        if args["sd"] != "":
            pickle.dump((dataset, labels), open(args["sd"], "wb"))
            print(f"saved dataset to {args['sd']}")

    print(f"{now()} loaded {len(dataset)} items in dataset with {len(labels)} labels")
    
    
    split = shared.split_dataset_BWR5(args["c"], sample_tracker, args["st"])
    print(
        f"{now()} split {len(split['train'])} training, "
        f"{len(split['validation'])} validation, and "
        f"{len(split['test'])} testing"
    )

    if args["z"] != "":
        dataset = shared.zero_dataset(dataset, args["z"])
        print(f"{now()} zeroed each item in dataset as data[{args['z']}]")

    model = DFNet(args["c"]) # one class for unmonitored
    if args["lm"] != "":
        model = torch.load(args["lm"])
        print(f"loaded model from {args['lm']}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"{now()} using {torch.cuda.get_device_name(0)}")
        model.cuda()

    if args["train"]:
        # Note below that shuffle=True is *essential*, 
        # see https://stackoverflow.com/questions/54354465/
        train_gen = data.DataLoader(
            shared.Dataset(split["train"], dataset, labels),
            batch_size=args["batchsize"], shuffle=True,
        )
        validation_gen = data.DataLoader(
            shared.Dataset(split["validation"], dataset, labels),
            batch_size=args["batchsize"], shuffle=True,
        )

        optimizer = torch.optim.Adamax(params=model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(args["epochs"]):
            print(f"{now()} epoch {epoch}")

            # training
            model.train()
            torch.set_grad_enabled(True)
            running_loss = 0.0
            n = 0
            for x, Y in train_gen:
                x, Y = x.to(device), Y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, Y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n+=1
            print(f"\ttraining loss {running_loss/n}")

            # validation
            model.eval()
            torch.set_grad_enabled(False)
            running_corrects = 0
            n = 0
            for x, Y in validation_gen:
                x, Y = x.to(device), Y.to(device)

                outputs = model(x)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == Y)
                n += len(Y)
            print(f"\tvalidation accuracy {float(running_corrects)/float(n)}")

        if args["sm"] != "":
            torch.save(model, args["sm"])
            print(f"saved model to {args['sm']}")
    
    # testing
    testing_gen = data.DataLoader(
        shared.Dataset(split["test"], dataset, labels), 
        batch_size=args["batchsize"]
    )
    model.eval()
    torch.set_grad_enabled(False)
    predictions = []
    p_labels = []
    for x, Y in testing_gen:
        x = x.to(device)
        outputs = model(x)
        index = F.softmax(outputs, dim=1).data.cpu().numpy()
        predictions.extend(index.tolist())
        p_labels.extend(Y.data.numpy().tolist())

    print(f"{now()} made {len(predictions)} predictions with {len(p_labels)} labels")
    csvline = []
    threshold = np.append([0], 1.0 - 1 / np.logspace(0.05, 2, num=15, endpoint=True))
    threshold = np.around(threshold, decimals=4)
    for th in threshold:
        tp, fpp, fnp, tn, fn, accuracy, recall, precision, f1 = shared.metrics(th, 
                                            predictions, p_labels, args["c"])
        print(
            f"\tthreshold {th:4.2}, "
            f"recall {recall:4.2}, "
            f"precision {precision:4.2}, "
            f"F1 {f1:4.2}, "
            f"accuracy {accuracy:4.2}   "
            f"[tp {tp:>5}, fpp {fpp:>5}, fnp {fnp:>5}, tn {tn:>5}, fn {fn:>5}]"
        )
        csvline.append([
            th, recall, precision, f1, accuracy, tp, fpp, fnp, tn, fn, args["extra"]
        ])

    if args["csv"]:
        with open(args["csv"], "w", newline="") as csvfile:
            w = csv.writer(csvfile, delimiter=",")
            w.writerow(["th", "recall", "precision", "f1", "accuracy", "tp", "fpp", "fnp", "tn", "fn", "extra"])
            w.writerows(csvline)
        print(f"saved testing results to {args['csv']}")

class DFNet(nn.Module):
    def __init__(self, classes, fc_in_features = 512*10):
        super(DFNet, self).__init__()
        # sources used when writing this, struggled with the change in output
        # size due to the convolutions and stumbled upon below:
        # - https://github.com/lin-zju/deep-fp/blob/master/lib/modeling/backbone/dfnet.py
        # - https://ezyang.github.io/convolution-visualizer/index.html
        self.kernel_size = 7
        self.padding_size = 3
        self.pool_stride_size = 4
        self.pool_size = 7

        self.block1 = self.__block(1, 32, nn.ELU())
        self.block2 = self.__block(32, 64, nn.ReLU())
        self.block3 = self.__block(64, 128, nn.ReLU())
        self.block4 = self.__block(128, 256, nn.ReLU())

        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.prediction = nn.Sequential(
            nn.Linear(512, classes),
            # when using CrossEntropyLoss, already computed internally
            #nn.Softmax(dim=1) # dim = 1, don't softmax batch
        )
    
    def __block(self, channels_in, channels, activation):
        return nn.Sequential(
            nn.Conv1d(channels_in, channels, self.kernel_size, padding=self.padding_size),
            nn.BatchNorm1d(channels),
            activation,
            nn.Conv1d(channels, channels, self.kernel_size, padding=self.padding_size),
            nn.BatchNorm1d(channels),
            activation,
            nn.MaxPool1d(self.pool_size, stride=self.pool_stride_size, padding=self.padding_size),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(start_dim=1) # dim = 1, don't flatten batch
        x = self.fc(x)
        x = self.prediction(x)

        return x

if __name__ == "__main__":
    main()