import csv
import sys
import jsonlines
from os.path import join

class InputExample(object):
    def __init__(self, text_a, text_b=None, label=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class DataProcessor(object):

    def get_dev_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        raise NotImplementedError()

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

    def _read_jsonl(cls, filepath):
        """ Reads the jsonl file path. """
        lines = []
        with jsonlines.open(filepath) as f:
            for line in f:
                lines.append(line)
        return lines


class SnliProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        filepath = join(data_dir, "train.tsv")
        items = self._read_tsv(filepath)
        return self._create_examples(items)

    def get_dev_examples(self, data_dir):
        filepath = join(data_dir, "dev.tsv")
        items = self._read_tsv(filepath)
        return self._create_examples(items)

    def get_test_examples(self, data_dir):
        filepath = join(data_dir, "test.tsv")
        items = self._read_tsv(filepath)
        return self._create_examples(items)

    def _create_examples(self, items):
        examples = []
        for (i, line) in enumerate(items):
            if i == 0:
                continue
            examples.append(InputExample(text_a=line[7], text_b=line[8]))
        return examples


class MnliProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        filepath = join(data_dir, "train.tsv")
        items = self._read_tsv(filepath)
        return self._create_examples(items)

    def get_dev_examples(self, data_dir):
        filepath = join(data_dir, "dev_matched.tsv")
        items = self._read_tsv(filepath)
        return self._create_examples(items)

    def get_test_examples(self, data_dir):
        filepath = join(data_dir, "test_matched.tsv")
        items = self._read_tsv(filepath)
        return self._create_examples(items)

    def _create_examples(self, samples):
        examples = []
        for (i, line) in enumerate(samples):
            if i == 0:
                continue
            s1 = line[8]
            s2 = line[9]
            examples.append(InputExample(text_a=s1, text_b=s2))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    def get_dev_examples(self, data_dir):
        filepath = join(data_dir, "dev_mismatched.tsv")
        items = self._read_tsv(filepath)
        return self._create_examples(items)

    def get_test_examples(self, data_dir):
        filepath = join(data_dir, "test_mismatched.tsv")
        items = self._read_tsv(filepath)
        return self._create_examples(items)



class NliProcessor(DataProcessor):
    def get_test_examples(self, data_dir):
        s1s = [line.rstrip() for line in open(join(data_dir, 's1.test'))]
        s2s = [line.rstrip() for line in open(join(data_dir, 's2.test'))]
        examples = []
        for s1, s2 in zip(s1s, s2s):
            examples.append(InputExample(text_a=s1, text_b=s2))
        return examples


class HansProcessor(DataProcessor):
    def get_test_examples(self, data_dir):
        path = join(data_dir, "heuristics_evaluation_set.txt")
        items = self._read_tsv(path)
        examples = []
        for (i, line) in enumerate(items):
            if i == 0:
                continue
            examples.append(InputExample(text_a=line[5], text_b=line[6]))
        return examples



class CoNLLReader:
    def __init__(self, file):
        """
        :param file: FileIO object
        """
        self.file = file

    def __iter__(self):
        return self

    def __next__(self):
        sent = self.readsent()
        if sent == []:
            raise StopIteration()
        else:
            return sent

    def readsent(self):
        """Assuming CoNLL-U format, where the columns are:
        ID FORM LEMMA UPOSTAG XPOSTAG FEATS HEAD DEPREL DEPS MISC"""
        sent = []
        row_str = self.file.readline().strip()
        while row_str != "":
            row = {}
            columns = row_str.split()
            row["ID"] = int(columns[0])
            row["FORM"] = columns[1]
            row["LEMMA"] = columns[2] if len(columns) > 2 else "_"
            row["UPOSTAG"] = columns[3] if len(columns) > 3 else "_"
            row["XPOSTAG"] = columns[4] if len(columns) > 4 else "_"
            row["FEATS"] = columns[5] if len(columns) > 5 else "_"
            row["HEAD"] = columns[6] if len(columns) > 6 else "_"
            row["DEPREL"] = columns[7] if len(columns) > 7 else "_"
            row["DEPS"] = columns[8] if len(columns) > 8 else "_"
            row["MISC"] = columns[9] if len(columns) > 9 else "_"
            sent.append(row)
            row_str = self.file.readline().strip()
        return sent

    def close(self):
        self.file.close()

processors = {
    "mnli": MnliProcessor,
    "mnlimicro": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "snli": SnliProcessor,
    "addonerte": NliProcessor,
    "snlihard": NliProcessor,
    "dpr": NliProcessor,
    "sprl":NliProcessor,
    "fnplus": NliProcessor,
    "joci": NliProcessor,
    "mpe": NliProcessor,
    "scitail": NliProcessor,
    "sick": NliProcessor,
    "glue": NliProcessor,
    "qqp": NliProcessor,
    "mnlimatchedharddev": NliProcessor,
    "mnlimismatchedharddev": NliProcessor,
    "hans": HansProcessor,
}

