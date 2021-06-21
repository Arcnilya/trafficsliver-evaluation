#!/usr/bin/env python3
import numpy as np
import os
import sys
from torch.utils import data
import pickle # JM: added pickle
from tqdm import tqdm

def metrics(threshold, predictions, labels, label_unmon):
    ''' Computes a range of metrics.
    For details on the metrics, see, e.g., https://www.cs.kau.se/pulls/hot/baserate/
    '''
    tp, fpp, fnp, tn, fn, accuracy, recall, precision, f1 = 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0

    for i in range(len(predictions)):
        label_pred = np.argmax(predictions[i])
        prob_pred = max(predictions[i])
        label_correct = labels[i]

        # we split on monitored or unmonitored correct label
        if label_correct != label_unmon:
            # either confident and correct,
            if prob_pred >= threshold and label_pred == label_correct:
                tp = tp + 1
            # confident and wrong monitored label, or
            elif prob_pred >= threshold and label_pred != label_unmon:
                fpp = fpp + 1
            # wrong because not confident or predicted unmonitored for monitored
            else:
                fn = fn + 1
        else:
            if prob_pred < threshold or label_pred == label_unmon: # correct prediction?
                tn = tn + 1
            elif label_pred < label_unmon: # predicted monitored for unmonitored
                fnp = fnp + 1
            else: # this should never happen
                sys.exit(f"this should never, wrongly labelled data for {label_pred}")

    if tp + fn + fpp > 0:
        recall = round(float(tp) / float(tp + fpp + fn), 4)
    if tp + fpp + fnp > 0:
        precision = round(float(tp) / float(tp + fpp + fnp), 4)

    if precision > 0 and recall > 0:
        f1 = round(2*((precision*recall)/(precision+recall)), 4)

    accuracy = round(float(tp + tn) / float(tp + fpp + fnp + fn + tn), 4)

    return tp, fpp, fnp, tn, fn, accuracy, recall, precision, f1


class Dataset(data.Dataset):
   def __init__(self, ids, dataset, labels):
       self.ids = ids
       self.dataset = dataset
       self.labels = labels

   def __len__(self):
       return len(self.ids)

   def __getitem__(self, index):
       ID = self.ids[index]
       return self.dataset[ID], self.labels[ID]

def load_dataset(
    mon_dir, unm_dir, 
    classes, partitions, samples,
    length, extract_func
    ):
    ''' Loads the dataset from disk into two dictionaries for data and labels.
    The dictionaries are indexed by sample ID. The ID encodes if its a monitored
    or unmonitored sample to make it easier to debug, as well as some info about
    the corresponding data file on disk. 
    This function works assumes the structure of the following dataset:
    - "top50-partitioned-reddit-levels-cirucitpadding" 
    '''
    data = {}
    labels = {}

    # load monitored data
    for c in range(0,classes):
        for p in range(0,partitions):
            site = c*10 + p
            for s in range(0,samples):
                ID = f"m-{c}-{p}-{s}"
                labels[ID] = c

                # file format is {site}-{sample}.trace
                fname = f"{site}-{s}.trace"
                with open(os.path.join(mon_dir, fname), "r") as f:
                    data[ID] = extract_func(f.read(), length)

    # load unmonitored data
    dirlist = os.listdir(unm_dir)
    # make sure we only load a balanced dataset
    dirlist = dirlist[:len(data)]
    for fname in dirlist:
        ID = f"u-{fname}"
        labels[ID] = classes # start from 0 for monitored
        with open(os.path.join(unm_dir, fname), "r") as f:
            data[ID] = extract_func(f.read(), length)

    return data, labels


def load_dataset_BWR5(dir_path, length, dt, _gaps, ds_size, extract_func):
    data = {}
    labels = {}
    sample_tracker = [0] * 100

    foo = True
    for i in range(ds_size):
        print(f"loading set {i+1}/{ds_size}")
        for website in tqdm(os.listdir(os.path.join(dir_path, str(i)))):
            for fname in os.listdir(os.path.join(dir_path, str(i), website)):
                if '.cell' in fname:  
                    with open(os.path.join(dir_path, str(i), website, fname), "r") as f:
                        trace = extract_func(f.read(), length, dt, _gaps)
                        label, sample = fname.split('.')[0].split('-')
                        #ID = f"m-{label}-0-{sample}"
                        ID = f"m-{label}-0-{sample_tracker[int(label)]}"
                        data[ID] = trace
                        labels[ID] = int(label)
                        sample_tracker[int(label)] += 1

    return data, labels, sample_tracker


def extract_BWR5(trace, length, dt, gaps):
    data = np.zeros((1, length), dtype=np.float32)
    trace = trace.split("\n")
    i = 0
    for n in range(len(trace)-1):
        if gaps and int(trace[n]) == 0:
            data[0][i] = 0.0
            i += 1
        elif dt and int(trace[n]) != 0:
            data[0][i] = np.float(int(trace[n]))
            i += 1
        elif int(trace[n]) > 0:
            data[0][i] = 1.0
            i += 1
        elif int(trace[n]) < 0:
            data[0][i] = -1.0
            i += 1
        if i >= length:
            break
        # else: found a zero and gaps is false
    return data


CIRCPAD_EVENT_NONPADDING_SENT = "circpad_cell_event_nonpadding_sent"
CIRCPAD_EVENT_NONPADDING_RECV = "circpad_cell_event_nonpadding_received"
CIRCPAD_EVENT_PADDING_SENT = "circpad_cell_event_padding_sent"
CIRCPAD_EVENT_PADDING_RECV = "circpad_cell_event_padding_received"
CIRCPAD_ADDRESS_EVENT = "connection_ap_handshake_send_begin"

def trace2cells(log, length, strip=True):
    ''' A fast specialised function to generate cells from a trace.
    Based on circpad_to_wf() in circpad-sim/common.py, but only for cells.
    '''
    data = np.zeros((1, length), dtype=np.float32)
    n = 0

    if strip:
        for i, line in enumerate(log):
            if CIRCPAD_ADDRESS_EVENT in line:
                log = log[i:]
                break

    s = log.split("\n")
    for line in s:
        # outgoing is positive
        if CIRCPAD_EVENT_NONPADDING_SENT in line or \
           CIRCPAD_EVENT_PADDING_SENT in line:
            data[0][n] = 1.0
            n += 1
        # incoming is negative
        elif CIRCPAD_EVENT_NONPADDING_RECV in line or \
           CIRCPAD_EVENT_PADDING_RECV in line:
            data[0][n] = -1.0
            n += 1
        
        if n == length:
            break

    return data


def split_dataset(
    classes, partitions, samples, fold, labels,
    ):
    '''Splits the dataset based on fold.
    The split is only based on IDs, not the actual data. The result is a 8:1:1
    split into training, validation, and testing.
    '''
    training = []
    validation = []
    testing = []

    # monitored, split by _partition_
    for c in range(0,classes):
        for p in range(0,partitions):
            for s in range(0,samples):
                ID = f"m-{c}-{p}-{s}"
                i = (p+fold) % partitions

                if i < partitions-2:
                    training.append(ID)
                elif i < partitions-1:
                    validation.append(ID)
                else:
                    testing.append(ID)

    # unmonitored
    counter = 0
    for k in labels.keys():
        if not k.startswith("u"):
            continue
        i = (counter+fold) % partitions
        if i < partitions-2:
            training.append(k)
        elif i < partitions-1:
            validation.append(k)
        else:
            testing.append(k)     
        counter += 1

    split = {}
    split["train"] = training
    split["validation"] = validation
    split["test"] = testing
    return split


def split_dataset_BWR5(
    classes, sample_tracker, sub_traces
    ):
    '''Splits the dataset based on fold.
    The split is only based on IDs, not the actual data. The result is a 8:1:1
    split into training, validation, and testing.
    '''
    training = []
    validation = []
    testing = []

    # monitored, split by _sample_
    for c in range(0,classes):
        for s in range(0,sample_tracker[c]):
            x = (s % sub_traces) // 5
            ID = f"m-{c}-0-{s}"
            if x < int(sub_traces/5*0.8):
                training.append(ID)
            elif x < int(sub_traces/5*0.9):
                validation.append(ID)
            else:
                testing.append(ID)

    split = {}
    split["train"] = training
    split["validation"] = validation
    split["test"] = testing
    return split


def zero_dataset(dataset, z):
    index = z.split(":")
    start = int(index[0])
    stop = int(index[1])
    data = np.zeros((stop-start), dtype=np.float32)
    for k, v in dataset.items():
        v[:,start:stop] = data
        dataset[k] = v
    return dataset