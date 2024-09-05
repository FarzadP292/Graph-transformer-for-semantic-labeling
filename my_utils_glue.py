# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
import jsonlines
from io import open
import numpy as np
import torch
import pickle5 as pickle
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from os.path import join

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,
                 s1_graph_labels=None, s1_graph=None,
                 s2_graph_labels=None, s2_graph=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            s1_graph: specifies the heads indices connected to each token of the sentence s1.
            s2_graph: specifies the heads indices connected to each token of the sentence s2.
            s1_graph_labels: specifies the relations between the tokens in s1 and their heads.
            s2_graph_labels: specifies the relations between the tokens in s2 and their heads.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.s1_graph_labels = s1_graph_labels
        self.s2_graph_labels = s2_graph_labels
        self.s1_graph = s1_graph
        self.s2_graph = s2_graph


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                 graph_labels=None,  graph=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.graph_labels = graph_labels
        self.graph = graph


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_dependency_labels(self):
        return ['<l>:mark', '<l>:dep', '<l>:aux', '<l>:nsubj', '<l>:ccomp', '<l>:pobj', '<l>:advcl', '<l>:cop', \
            '<l>:root', '<l>:[clip]', '<l>:poss', '<l>:cc', '<l>:prep', '<l>:punct', '<l>:advmod', '<l>:dobj', \
            '<l>:pcomp', '<l>:xcomp', '<l>:unk', '<l>:amod', '<l>:conj']

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def __init__(self, graph_dir=None):
        self.num_classes = 3
        self.graph_dir = graph_dir

    def get_dependency_labels(self):
        return ['<l>:I-R-ARGM-TMP',
 '<l>:I-R-ARG1',
 '<l>:B-ARG2',
 '<l>:B-C-ARG0',
 '<l>:B-ARGM-NEG',
 '<l>:[CLIP]',
 '<l>:B-ARGM-COM',
 '<l>:B-ARGM-CAU',
 '<l>:B-R-ARGM-LOC',
 '<l>:I-ARG3',
 '<l>:I-C-ARG3',
 '<l>:I-R-ARG4',
 '<l>:B-ARGM-LOC',
 '<l>:I-C-ARG2',
 '<l>:I-R-ARGM-COM',
 '<l>:I-ARGM-ADV',
 '<l>:I-ARGM-CAU',
 '<l>:I-V',
 '<l>:B-R-ARG2',
 '<l>:I-ARGM-ADJ',
 '<l>:B-C-ARGM-LOC',
 '<l>:I-ARGM-LOC',
 '<l>:I-ARGM-MNR',
 '<l>:B-R-ARGM-COM',
 '<l>:I-R-ARGM-LOC',
 '<l>:B-ARG4',
 '<l>:I-C-ARGM-EXT',
 '<l>:I-R-ARGM-MNR',
 '<l>:B-R-ARGM-EXT',
 '<l>:I-ARG0',
 '<l>:B-R-ARGM-ADV',
 '<l>:I-R-ARGM-DIR',
 '<l>:I-ARGM-PRP',
 '<l>:B-ARGM-TMP',
 '<l>:I-ARGM-DIS',
 '<l>:B-C-ARG4',
 '<l>:I-ARGM-PNC',
 '<l>:I-R-ARG3',
 '<l>:B-C-ARGM-ADV',
 '<l>:B-ARG3',
 '<l>:I-ARG5',
 '<l>:B-C-ARGM-MNR',
 '<l>:I-C-ARGM-ADV',
 '<l>:I-C-ARG4',
 '<l>:B-ARGM-ADV',
 '<l>:B-C-ARG3',
 '<l>:B-C-ARG2',
 '<l>:B-ARGM-DIS',
 '<l>:B-C-ARG1',
 '<l>:I-R-ARG2',
 '<l>:B-ARGM-DIR',
 '<l>:B-R-ARG4',
 '<l>:I-C-ARGM-TMP',
 '<l>:B-R-ARGM-TMP',
 '<l>:B-ARGM-PRD',
 '<l>:B-ARGM-PRR',
 '<l>:I-C-ARGM-MNR',
 '<l>:I-R-ARGM-CAU',
 '<l>:B-ARGM-REC',
 '<l>:Kossher',
 '<l>:B-ARGM-PNC',
 '<l>:I-ARGM-DIR',
 '<l>:B-ARGM-GOL',
 '<l>:B-R-ARGM-DIR',
 '<l>:I-ARG4',
 '<l>:B-ARGM-LVB',
 '<l>:B-ARGM-EXT',
 '<l>:I-R-ARGM-EXT',
 '<l>:B-ARGM-MOD',
 '<l>:B-R-ARGM-MNR',
 '<l>:B-R-ARG0',
 '<l>:B-ARGA',
 '<l>:B-ARGM-PRP',
 '<l>:B-V',
 '<l>:B-ARG5',
 '<l>:I-ARGM-PRD',
 '<l>:I-ARGM-GOL',
 '<l>:B-ARGM-ADJ',
 '<l>:I-R-ARGM-GOL',
 '<l>:B-ARG1',
 '<l>:I-ARGM-MOD',
 '<l>:I-ARGM-TMP',
 '<l>:B-R-ARGM-GOL',
 '<l>:I-C-ARG1',
 '<l>:I-ARGM-EXT',
 '<l>:B-C-ARGM-EXT',
 '<l>:I-ARGM-NEG',
 '<l>:I-R-ARGM-ADV',
 '<l>:I-C-ARG0',
 '<l>:B-R-ARGM-CAU',
 '<l>:B-ARG0',
 '<l>:B-C-ARGM-TMP',
 '<l>:I-C-ARGM-LOC',
 '<l>:I-ARGM-COM',
 '<l>:B-ARGM-MNR',
 '<l>:B-R-ARG3',
 '<l>:I-R-ARG0',
 '<l>:I-ARG2',
 '<l>:B-R-ARG1',
 '<l>:I-ARG1',
 '<l>:root',
         '<l>:unk']
    

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        if self.graph_dir is not None:
            s1_graph_data = pickle.load(
                open(os.path.join(self.graph_dir, "s1_parse." + set_type + ".graph.pkl"), "rb"))
            s2_graph_data = pickle.load(
                open(os.path.join(self.graph_dir, "s2_parse." + set_type + ".graph.pkl"), "rb"))

        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            elif i== 11:
                break
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            if self.graph_dir is None:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            else:
                text_a = s1_graph_data[i-1][0]
                text_b = s2_graph_data[i-1][0]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                                                 s1_graph_labels=s1_graph_data[i-1][1],
                                                 s1_graph=s1_graph_data[i-1][2],
                                                 s2_graph_labels=s2_graph_data[i-1][1],
                                                 s2_graph=s2_graph_data[i-1][2]))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_mismatched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 1

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["not_entailment", "entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def __init__(self):
        self.num_classes = 2

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SnliProcessor(DataProcessor):
    """Processor for the SNLI data set (GLUE version)."""
    def __init__(self, graph_dir=None):
        self.num_classes = 3
        self.graph_dir = graph_dir

    def get_dependency_labels(self):
        return ['<l>:unk', '<l>:punct', '<l>:dep', '<l>:aux', '<l>:det', '<l>:advcl', '<l>:pobj', '<l>:root',\
                '<l>:advmod', '<l>:nsubj', '<l>:mark', '<l>:nn', '<l>:dobj', '<l>:prep', '<l>:[clip]', '<l>:bert_subtoken']

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""

        if self.graph_dir is not None:
            s1_graph_data = pickle.load(
                open(os.path.join(self.graph_dir, "s1_parse."+set_type+".graph.pkl"), "rb"))
            s2_graph_data = pickle.load(
                open(os.path.join(self.graph_dir, "s2_parse."+set_type+".graph.pkl"), "rb"))

        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]

            if self.graph_dir is None:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            else:
                text_a = s1_graph_data[i-1][0]
                text_b = s2_graph_data[i-1][0]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                                             s1_graph_labels=s1_graph_data[i-1][1],
                                             s1_graph=s1_graph_data[i-1][2],
                                             s2_graph_labels=s2_graph_data[i-1][1],
                                             s2_graph=s2_graph_data[i-1][2]))
        return examples


class NliProcessor(DataProcessor):
    """Processor for the dataset of the format of SNLI
    (InferSent version), could be 2 or 3 classes."""

    def __init__(self, data_dir, graph_dir=None):
        # We assume there is a training file there and we read labels from there.
        labels = [line.rstrip() for line in open(join(data_dir, 'labels.train'))]
        self.labels = list(set(labels))
        labels = ["contradiction", "entailment", "neutral"]
        ordered_labels = []
        for l in labels:
            if l in self.labels:
                ordered_labels.append(l)
        self.labels = ordered_labels
        self.num_classes = len(self.labels)
        self.graph_dir = graph_dir

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        if self.graph_dir is not None:
            s1_graph_data = pickle.load(
                open(os.path.join(self.graph_dir, "s1_parse."+set_type+".graph.pkl"), "rb"))
            s2_graph_data = pickle.load(
                open(os.path.join(self.graph_dir, "s2_parse."+set_type+".graph.pkl"), "rb"))

        s1s = [line.rstrip() for line in open(join(data_dir, 's1.' + set_type))]
        s2s = [line.rstrip() for line in open(join(data_dir, 's2.' + set_type))]
        labels = [line.rstrip() for line in open(join(data_dir, 'labels.' + set_type))]

        examples = []
        for (i, line) in enumerate(s1s):
            guid = "%s-%s" % (set_type, i)
            text_a = s1s[i]
            text_b = s2s[i]
            label = labels[i]
            # In case of hidden labels, changes it with entailment.
            if label == "hidden":
                label = "entailment"
            if self.graph_dir is None:
               examples.append(
                  InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            else:
               text_a = s1_graph_data[i][0]
               text_b = s2_graph_data[i][0]
               examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                                             s1_graph_labels=s1_graph_data[i][1],
                                             s1_graph=s1_graph_data[i][2],
                                             s2_graph_labels=s2_graph_data[i][1],
                                             s2_graph=s2_graph_data[i][2]))
        return examples


class HansProcessor(DataProcessor):
    """Processor for the processed Hans dataset."""
    def __init__(self, graph_dir=None):
        self.num_classes = 2
        self.graph_dir = graph_dir 

    def read_jsonl(self, filepath):
        """ Reads the jsonl file path. """
        lines = []
        with jsonlines.open(filepath) as f:
           for line in f:
             lines.append(line)
        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        pass

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(\
            self._read_tsv(join(data_dir, "heuristics_evaluation_set.txt")), \
            "dev")

    def get_dev_labels(self, data_dir):
        items = self._read_tsv(os.path.join(data_dir, "heuristics_evaluation_set.txt"))
        labels = []
        for (i, item) in enumerate(items):
            if i == 0:
               continue
            label = items[i][0]
            labels.append(label)
        return np.array(labels)

    def get_labels(self):
        """See base class."""
        return ["non-entailment", "entailment"]

    def _create_examples(self, items, set_type):
        """Creates examples for the training and dev sets."""
        if self.graph_dir is not None:
            s1_graph_data = pickle.load(
                open(os.path.join(self.graph_dir, "s1_parse.test.graph.pkl"), "rb"))
            s2_graph_data = pickle.load(
                open(os.path.join(self.graph_dir, "s2_parse.test.graph.pkl"), "rb"))
        
        examples = []
        for (i, item) in enumerate(items):
            if i == 0:
               continue
            guid = "%s-%s" % (set_type, i)
            # Claim has artifacts so this needs to be text_b.
            text_a = items[i][5]
            text_b = items[i][6]
            label = items[i][0]

            if self.graph_dir is None:
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            else:
                text_a = s1_graph_data[i-1][0]
                text_b = s2_graph_data[i-1][0]
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,
                                s1_graph_labels=s1_graph_data[i-1][1],
                                s1_graph=s1_graph_data[i-1][2],
                                s2_graph_labels=s2_graph_data[i-1][1],
                                s2_graph=s2_graph_data[i-1][2]))

        return examples

# TODO: now CLS is the first root and second root is SEP, we may want to change this later.
def join_graphs(example,max_length):
    # Considers for the root node to be connected to themselves.
    
    
    assert len(example.s1_graph) + len(example.s2_graph) + 2 <= max_length
    s1_graph = [[0]]+example.s1_graph 
    
    s2_graph = [[0]] + example.s2_graph
    
    for  i, val in enumerate(s2_graph):
        s2_graph[i] = [x + len(example.s1_graph)+1  for x in val]
    
    joined_graph = s1_graph + s2_graph# --------here------------------
    
    # The heads greater than the context_size are replaced with max_length and after the offset they
    # become larger than context size, so we need to clip this again here.
    max_length_new = len(example.s1_graph) + len(example.s2_graph) + 2
    
    for i, val in enumerate(joined_graph):
        joined_graph[i] = [min(element, max_length_new) for element in val]# --------here------------------
    
    assert all(x_ < max_length for x in joined_graph for x_ in x)
    
    joined_graph_labels = [['root']]+example.s1_graph_labels+[['root']]+example.s2_graph_labels
    return joined_graph_labels, joined_graph


def equalize_listSize(L, fill):
    
    M = max([len(l) for l in L])
    
    
    for i, val in enumerate(L):
        L[i].extend([fill]*(M-len(val)))
    
    return L, M

    

# TODO: we need to decide for the pad graph ids.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False, pad_on_left=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,
                                 mask_padding_with_zero=True, with_graph=False,
                                 pad_graph_labels_token=0, pad_graph_token=-1):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    semantic_labels = []# ----------------------------------------here-------------------------------------------
    ccc = 0
    total_ccc = 0
    features = []
    max_deg = 0
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if with_graph:
            tokens_a = example.text_a
        else:
            tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            if with_graph:
                tokens_b = example.text_b
            else:
                tokens_b = tokenizer.tokenize(example.text_b)

            # We assume truncation is done in the preprocessing.
            if with_graph:
                assert len(tokens_a)+len(tokens_b) <= max_seq_length - 3, "tokens_a:{},tokens_b:{}".format(len(tokens_a),len(tokens_b))
                assert len(tokens_a) == len(example.s1_graph_labels) == len(example.s1_graph),\
                    "The parsing length does not match the token length for s1 sentence: words" \
                    ":{}, labels:{}, graph:{}, org_text:{} ".format(
                        tokens_a,example.s1_graph_labels,example.s1_graph,example.text_a)
                assert len(tokens_b) == len(example.s2_graph_labels) == len(example.s2_graph),\
                    "The parsing length does not match the token length for s1 sentence: words" \
                     ":{}, labels:{}, graph:{}, org_text:{} ".format(
                        tokens_b,example.s2_graph_labels,example.s2_graph,example.text_b)
            else:
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        if with_graph:
            joined_graph_labels, joined_graph = join_graphs(example,max_seq_length)#------------------here-----------------------------
            
            tokenized_joined_graph_label = [None]*len(joined_graph_labels)
            
            for i, val in enumerate(joined_graph_labels):#------------------here-----------------------------
                tokenized_joined_graph_label[i] =   [tokenizer.tokenize('<l>:'+label)[0] if len(tokenizer.tokenize('<l>:'+label))==1
                                                else "<l>:unk" for label in val]
                
            #------------------here-----------------------------
            joined_graph_labels_ids=[None]*len(tokenized_joined_graph_label)
            for i, val in enumerate(tokenized_joined_graph_label):
              for x in val:
                if x == "<l>:unk":
                    ccc += 1.0
                total_ccc += 1.0
                joined_graph_labels_ids[i] = tokenizer.convert_tokens_to_ids(val)

            # Since 3 tokens is considered to reach the max_length, adding two
            # roots is still below the max number of tokens.
            graph_padding_length = max_seq_length - len(joined_graph_labels_ids)
            #------------------here-----------------------------
            joined_graph_labels_ids, M1 = equalize_listSize(joined_graph_labels_ids, -100)
            joined_graph, M2 = equalize_listSize(joined_graph, -100)
            assert M1==M2 
            
            if M1 > max_deg:
                max_deg = M1
            
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids

            if with_graph: #------------------here-----------------------------
                pad_graph_labels_token = tokenizer.convert_tokens_to_ids("<l>:unk")
                joined_graph_labels_ids = [[pad_graph_labels_token]*M1]*graph_padding_length + joined_graph_labels_ids
                joined_graph = [[pad_graph_token]*M2]*graph_padding_length +joined_graph
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            if with_graph:#------------------here-----------------------------
                pad_graph_labels_token = tokenizer.convert_tokens_to_ids("<l>:unk")
                joined_graph_labels_ids = joined_graph_labels_ids + [[pad_graph_labels_token]*M1]*graph_padding_length
                joined_graph = joined_graph + [[pad_graph_token]*M2]*graph_padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if with_graph:
           assert len(joined_graph) == max_seq_length
           assert len(joined_graph_labels_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        if with_graph:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              graph_labels=joined_graph_labels_ids,
                              graph=joined_graph))
        else:
            features.append(
               InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    #print("Percentage of UNK labels:{}".format(ccc*100.0/total_ccc))
    return features,max_deg


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def per_class_accuracy(preds, labels):
    unique_labels = np.unique(labels)
    results = {}
    for l in unique_labels:
        indices = (l == labels)
        acc = (preds[indices] == labels[indices]).mean()
        results["acc_" + str(int(l))] = acc
    acc = (preds == labels).mean()
    results["acc"] = acc
    return results


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name in ["mnli", "mnli-mm", "qnli", "rte", "wnli", "snli", "nli", "mnlimicro","hans", \
                       "HANS-const", "HANS-lex", "HANS-sub", "sst-2", "mnlitest"]:
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "snli": SnliProcessor,
    "nli": NliProcessor,
    "hans":  HansProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnlitest": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "snli": "classification",
    "nli": "classification",
    "hans": "classification",
    "HANS-const": "classification",
    "HANS-lex": "classification",
    "HANS-sub": "classification"
}

GLUE_TASKS_NUM_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mnli-mm": 3,
    "mrpc": 2,
    "sst-2": 2,
    "sts-b": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
    "snli": 3,
    "hans": 2,
    "HANS-const": 2,
    "HANS-lex": 2,
    "HANS-sub": 2,
}
