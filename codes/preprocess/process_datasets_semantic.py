import argparse
import os
import pickle
from os.path import join
from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
from transformers import BertTokenizer
from data import processors
from transformers import AutoTokenizer



# TODO: think of removing bert_tokens_lengths
class ParsingExample(object):
    def __init__(self, sentence, bert_tokens, lengths, tokens=None, heads=None, labels=None):
        self.sentence = sentence
        self.bert_tokens = bert_tokens
        self.lengths = lengths
        self.tokens = tokens
        self.heads = heads
        self.labels = labels

    def __repr__(self):
        return "Sentence:{},Tokens:{}, BERT Tokens:{}, Heads:{}, Labels:{}" \
            .format(self.sentence,self.tokens, self.bert_tokens, self.heads, self.labels)

class DependencyParser:
    def __init__(self, max_length, do_truncate, tokenizer_path='pretrained_transformers/bert-base-uncased',
                 do_lower_case=True):
        # self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=do_lower_case)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        self.max_length = max_length
        self.do_truncate = do_truncate

    def truncate_pairs_wordlevel(self, s1, s2):
        max_len = self.max_length - 3
        while True:
            if sum(s1.lengths) + sum(s2.lengths) <= max_len:
                break
            if sum(s1.lengths) > sum(s2.lengths):
                s1.tokens.pop()
                s1.lengths.pop()
                s1.bert_tokens.pop()
                if s1.labels is not None:  # parsing information is computed beforehand.
                    s1.labels.pop()
                    s1.heads.pop()
            else:
                s2.tokens.pop()
                s2.lengths.pop()
                s2.bert_tokens.pop()
                if s2.labels is not None:
                    s2.labels.pop()
                    s2.heads.pop()

    def write_to_file(self, data, output_dir):
        with open(output_dir, 'w') as f:
            for d in data:
                f.write(d + "\n")

    def get_examples(self, task, type, data_dir):
        processor = processors[task]()
        if type == "train":
            return processor.get_train_examples(data_dir)
        elif "test" in type:
            return processor.get_test_examples(data_dir)
        elif "dev" in type:
            return processor.get_dev_examples(data_dir)
        
    def get_heads(self,parsed):
    
        W = parsed['words']
    
        V = parsed['verbs']
    
        list_of_heads = [[] for _ in range(len(W))]
        list_of_labels = [[] for _ in range(len(W))]
    
        for verb in V:
        
            i = verb['tags'].index('B-V')
            list_of_heads[i].append(0)
            list_of_labels[i].append('root')
            
            for j,tag in enumerate(verb['tags']):
                if tag != 'O':
                    if j!=i :
                        list_of_heads[j].append(i+1)
                        list_of_labels[j].append(tag)

        return list_of_heads, list_of_labels

    def extract_tokens_parse_info(self, sentence):
        # source: https://demo.allennlp.org/dependency-parsing
        results = self.predictor.predict(sentence=sentence)
        tokens = results['words']

        ## tokenize each word with bert
        tokens_bert = [self.tokenizer.tokenize(word) if len(self.tokenizer.tokenize(word)) > 0
                          else self.tokenizer.tokenize('[UNK]') for word in tokens]
        tokens_length = [len(token) for token in tokens_bert]

        assert all(tokens_length), "BERT tokens before truncating:{}".format(tokens_bert_s1)
        # assert len(results["predicted_heads"]) == len(results["predicted_dependencies"]) \
        #        == len(tokens), \
        #     "Heads:{},Labels:{},Words:{}".format(results["predicted_heads"],
        #                                          results["predicted_dependencies"], tokens)
        heads, labels = self.get_heads(results)
        example = ParsingExample(sentence,tokens_bert,tokens_length,tokens,
                                 heads,labels)
        return example

    def clip_heads(self, s1_info, s2_info):
        # we connect all the truncated tokens to token=SEP2, which is token with index = len1+len2+2
        # since data is in the format of CLS+s1+SEP+s2+SEP
        truncated_head_index = sum([len(x) for x in s1_info.bert_tokens])+\
                               sum([len(x) for x in s2_info.bert_tokens])+2
                               
        for i,items in  enumerate(s1_info.heads):
            h_temp=items
            l_temp=s1_info.labels[i]
            for j,item in enumerate(items):
                
                if item > len(s1_info.bert_tokens):
                    h_temp[j]=truncated_head_index
                    l_temp[j]="[CLIP]"
                else:
                    h_temp[j]=item
                    l_temp[j]= l_temp[j]
                
            s1_info.heads[i] = h_temp
            s1_info.labels[i] = l_temp
            
        
        for i,items in  enumerate(s2_info.heads):
            h_temp=items
            l_temp=s2_info.labels[i]
            for j,item in enumerate(items):
                
                if item > len(s2_info.bert_tokens):
                    h_temp[j]=truncated_head_index
                    l_temp[j]="[CLIP]"
                else:
                    h_temp[j]=item
                    l_temp[j]= l_temp[j]
                
            s2_info.heads[i] = h_temp
            s2_info.labels[i] = l_temp
            

        return s1_info, s2_info

    def extract_parses(self, dataset, output_dir, option, type, data_dir):
        examples = self.get_examples(dataset, type, data_dir)
        s1_parses = []
        s2_parses = []
        for i, example in enumerate(examples):
            
            if i==10:
                break 
            
            s1_info = self.extract_tokens_parse_info(example.text_a)
            s2_info = self.extract_tokens_parse_info(example.text_b)

            if self.do_truncate == "after_parse":
                self.truncate_pairs_wordlevel(s1_info, s2_info)
                s1_info, s2_info = self.clip_heads(s1_info, s2_info)
            s1_parses.append(s1_info)
            s2_parses.append(s2_info)
        truncated_head_indices = [sum([len(x) for x in s1.bert_tokens])+sum([len(x) for x in s2.bert_tokens])
                                  +2 for s1, s2 in zip(s1_parses, s2_parses)]
        outfile = join(output_dir, "s1_parse." + type + ".graph")
        self.expand_bert_subtokens_graph(s1_parses, outfile, option, truncated_head_indices)
        outfile = join(output_dir, "s2_parse." + type + ".graph")
        self.expand_bert_subtokens_graph(s2_parses, outfile, option, truncated_head_indices)

    def extract(self, data_dir, output_dir, dataset, option):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if dataset == "snli":
            for type in ["train", "dev", "test"]:
                self.extract_parses(dataset, output_dir, option, type, data_dir)
        elif dataset in ["mnli", "mnlimicro", "mnlitest"]:
            for type in ["train", "dev_matched", "test_matched"]:
                self.extract_parses("mnli", output_dir, option, type, data_dir)
            for type in ["dev_mismatched", "test_mismatched"]:
                    self.extract_parses("mnli-mm", output_dir, option, type, data_dir)
        elif dataset == "hans":
            for type in ["constituent", "lexical_overlap", "subsequence"]:
                data_dir_type = join(data_dir, type)
                output_dir_type = join(output_dir, type)
                if not os.path.exists(output_dir_type):
                    os.makedirs(output_dir_type)
                self.extract_parses(dataset, output_dir_type, option, "test", data_dir_type)

        elif dataset in args.nli_datasets:
            self.extract_parses(dataset, output_dir, option, "test", data_dir)

    def get_expanded_heads_labels(self, tokens_length, heads, labels, option, truncated_head_index):
        # Builds a mapping from the old node index to the new node index.
        old_to_new_node = {0: [0]}
        assert all(tokens_length), "BERT token lengths".format(tokens_length)
        index = 1
        for token_id, token_length in enumerate(tokens_length):
            if token_length == 1:
                old_to_new_node[token_id + 1] = [index]
                index += token_length
            else:
                old_to_new_node[token_id + 1] = [index+i for i in range(token_length)]
                index += token_length
        # Builds the expanded heads and labels.

        index = 1

        expanded_heads = []
        expanded_labels = []
        
        for token_id, token_length in enumerate(tokens_length):
            h_temp=[]
            l_temp=[]
            for i,h in enumerate(heads[token_id]):
                if h == truncated_head_index:
                    h_temp.append(h)
                    l_temp.append(labels[token_id][i])
                else:

                    h_temp.extend(old_to_new_node[h])
                    l_temp.extend([labels[token_id][i]]*len(old_to_new_node[h]))

                    
                    
            expanded_heads.append(h_temp)
            
            expanded_labels.append(l_temp)
            
            for sub_token in range(token_length - 1):
                if option == "connect_to_first_token":
                    expanded_heads.append(index)
                    expanded_labels.append("bert_subtoken")
                elif option == "connect_to_first_token_head":
                    expanded_heads.append(h_temp)

                    expanded_labels.append(l_temp)
                    
                elif option == "connect_to_null":
                    expanded_heads.append(-1)
                    expanded_labels.append("bert_subtoken")
            index += token_length
        assert sum(tokens_length) == len(expanded_heads) == len(expanded_labels)
        return expanded_heads, expanded_labels

    def expand_bert_subtokens_graph(self, examples, outfile, option, truncated_head_indices):
        words_labels_heads = []
        for i, example in enumerate(examples):
            expanded_heads, expanded_labels = self.get_expanded_heads_labels(example.lengths, example.heads,
                example.labels, option, truncated_head_indices[i])
            bert_tokens = [item for sublist in example.bert_tokens for item in sublist]
            assert len(bert_tokens) == len(expanded_heads) == len(expanded_labels), \
                "Bert tokens:{},Head vector:{},Label vector:{}".format(example.bert_tokens, expanded_heads,
                                                                       expanded_labels)

            temp_heads = expanded_heads
            expanded_heads=[x if len(x)!=0 else [-2000] for x in temp_heads]
        
            temp_labels = expanded_labels
            expanded_labels=[x if len(x)!=0 else ['Kossher'] for x in temp_labels]
                
            words_labels_heads.append((bert_tokens, expanded_labels, expanded_heads))
        
        with open(outfile + '.pkl', 'wb') as f:
            pickle.dump(words_labels_heads, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq", type=int, default=128)
    parser.add_argument('--do_truncate', type=str, choices=[None, "before_parse", "after_parse"], default="after_parse",
                        help='Truncate for the specified dataset before or after computing the parsing info')
    parser.add_argument("--outputdir", type=str, default="./")
    parser.add_argument('--dataset', type=str,
                        choices=["snli", "mnli", "mnlimicro", "mnli-mm", "snlihard", "hans", "addonerte", "dpr", "sprl",
                        "fnplus", "mnlitest", "joci", "mpe", "scitail", "sick", "glue", "qqp", "mnlimatchedharddev",
                        "mnlimismatchedharddev"], help="Name of the dataset", default="mnli")
    parser.add_argument("--option", type=str, choices=["connect_to_first_token", "connect_to_null", \
                                                       "connect_to_first_token_head"],
                        default="connect_to_first_token_head")
    args = parser.parse_args()

    parser = DependencyParser(args.max_seq, args.do_truncate)

    args.task_to_data_dir = {
        "snli": "./datasets/SNLI",
        "snli-hard":  "./datasets/SNLIHard",
        "mnli": "./datasets/MNLI",
        "mnli-mm": "./datasets/MNLI",
        "mnlimicro": "./datasets/MNLImicro",
        "snlihard": "./datasets/SNLIHard",
        "hans": "./datasets/hans",
        "addonerte": "./datasets/AddOneRTE",
        "dpr": "./datasets/DPR/",
        "sprl": "./datasets/SPRL/",
        "fnplus": "./datasets/FNPLUS/",
        "joci": "./datasets/JOCI/",
        "mpe": "./datasets/MPE/",
        "scitail": "./datasets/SciTail/",
        "sick": "./datasets/SICK/",
        "glue": "./datasets/GLUEDiagnostic/",
        "qqp": "./datasets/QQP/",
        "mnlimismatchedharddev": "./datasets/MNLIMismatchedHardWithHardTest",
        "mnlimatchedharddev": "./datasets/MNLIMatchedHardWithHardTest/",
        "mnlitest": "./datasets/SentEval/data/GLUE/MNLItest",

    }
    args.nli_datasets = ["addonerte", "dpr", "sprl", "fnplus", "joci", "mpe", "snlihard",
                         "scitail", "sick", "glue", "qqp", "mnlimatchedharddev", "mnlimismatchedharddev"]
    data_dir = args.task_to_data_dir[args.dataset]
    
    out_dir = join(args.outputdir, "nli_parsing/semantic/" + args.dataset)
    if args.do_truncate:
        out_dir = out_dir + "_truncate_" + str(args.max_seq)
    parser.extract(data_dir, out_dir, args.dataset, args.option)
