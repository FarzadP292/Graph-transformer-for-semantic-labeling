""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import json
import argparse
import glob
import logging
import os
import random
from utils_glue import GLUE_TASKS_NUM_LABELS

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from mutils import write_to_csv
from os.path import join

# TODO: we need to also update the run_glue from the new transformer files.

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer,
                          XLMConfig, XLMForSequenceClassification,
                          XLMTokenizer, XLNetConfig,
                          XLNetForSequenceClassification,
                          XLNetTokenizer,
                          get_linear_schedule_with_warmup)

from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP

from my_utils_bert import BertGraphForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from my_utils_glue import (compute_metrics, convert_examples_to_features,
                        processors)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)




ALL_MODELS = sum((tuple(conf.keys()) for conf in (BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLM_PRETRAINED_CONFIG_ARCHIVE_MAP)),
                  ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertGraphForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def save_model(args, global_step, model, logger):
    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    logger.info("Saving model checkpoint to %s", output_dir)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.use_two_optimizer:
        model_nonbert = []
        model_bert = []
        layernorm_params = ['layernorm_key_layer', 'layernorm_value_layer', 'dp_relation_k', 'dp_relation_v']
        for name, param in model.named_parameters():
            if args.add_classifier_to_bert_optimizer:
                if not any(nd in name for nd in layernorm_params):
                    model_bert.append((name, param))
                else:
                    model_nonbert.append((name, param))
            else:
                if 'bert' in name and not any(nd in name for nd in layernorm_params):
                    model_bert.append((name, param))
                else:
                    model_nonbert.append((name, param))

        # Prepare optimizer and schedule (linear warmup and decay) for Non-bert parameters
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters_nonbert = [
            {'params': [p for n, p in model_nonbert if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model_nonbert if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer_nonbert = AdamW(optimizer_grouped_parameters_nonbert, lr=args.lr_nonbert, eps=args.adam_epsilon)
        scheduler_nonbert = get_linear_schedule_with_warmup(
            optimizer_nonbert, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )


        # Prepare optimizer and schedule (linear warmup and decay) for Bert parameters
        optimizer_grouped_parameters_bert = [
            {'params': [p for n, p in model_bert if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model_bert if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer_bert = AdamW(optimizer_grouped_parameters_bert, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler_bert= get_linear_schedule_with_warmup(
            optimizer_bert, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            if args.with_graph:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3],
                          'graph_labels': batch[4],
                          'graph': batch[5],
                          }
            else:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3],
                          }

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.use_two_optimizer:
                    scheduler_bert.step()
                    scheduler_nonbert.step()
                    optimizer_bert.step()
                    optimizer_nonbert.step()
                else:
                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    if args.use_two_optimizer:
                        tb_writer.add_scalar('lr', scheduler_bert.get_lr()[0], global_step)
                        tb_writer.add_scalar('lr_nonbert', scheduler_nonbert.get_lr()[0], global_step)
                    else:
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    save_model(args, global_step, model, logger)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()
    return global_step, tr_loss / global_step


# writes the labels in the kaggle format.
def write_in_kaggle_format(args, label_ids, gold_labels, save_labels_file, eval_task):
    # make a dictionary from the labels.
    labels_map = {}
    i = 0
    for label in gold_labels:
        labels_map[i] = label
        i = i + 1

    ids_file = join(args.task_to_data_dir[eval_task], "ids.test")
    ids = [line.strip('\n') for line in open(ids_file)]

    with open(save_labels_file, 'w') as f:
        f.write("pairID,gold_label\n")
        for i, l in enumerate(label_ids):
            label = labels_map[l]
            f.write("{0},{1}\n".format(ids[i], label))


def binarize_preds(preds):
    # maps the third label (neutral one) to first, which is contradiction.
    preds[preds == 2] = 0
    return preds


def binerize_sum_preds(preds):
    binerize_preds = np.stack([
        preds[:, 0] + preds[:, 2],
        preds[:, 1]
    ], 1)
    return binerize_preds


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    if args.task_name == "mnli" and "mnli-mm" not in args.task_name:
        args.eval_task_names.append("mnli-mm")

    results = {}
    for eval_task in args.eval_task_names:
        # eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_labels, num_classes = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        args.eval_batch_size = args.per_gpu_eval_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                # TODO: do we also need graphs during the evaluation? If we dont then this is a more powerful method.
                if args.with_graph:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              'labels': batch[3],
                              'graph_labels': batch[4],
                              'graph': batch[5],
                              }
                else:
                    inputs = {'input_ids': batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                              # XLM don't use segment_ids
                              'labels': batch[3]}

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if num_classes == 2 and args.binerize_sum:
            preds = binerize_sum_preds(preds)

        preds = np.argmax(preds, axis=1)

        # convert 1,2 labels to 1 in case of binary dataset.
        if num_classes == 2 and args.binerize_eval:
            preds = binarize_preds(preds)

        if eval_task in args.nli_task_names:
            eval_task_metric = "nli"
        elif eval_task == "mnlismall":
            eval_task_metric = "mnli"
        else:
            eval_task_metric = eval_task

        result = compute_metrics(eval_task_metric, preds, out_label_ids)

        if args.save_labels_file is not None:
            save_labels_file = args.save_labels_file + "_" + eval_task
            write_in_kaggle_format(args, preds, eval_labels, save_labels_file, eval_task)

        results[eval_task] = result["acc"]
        print("results is ", result, " eval_task ", eval_task)

    return results


def get_graph_dir(args, task):
    graph_dir = args.task_to_graph_dir[task]
    if args.do_truncate: 
        if "hans" in graph_dir:
           splits = graph_dir.split("/")
           first_part = splits[:-1]
           second_part = splits[-1]
           graph_dir = "/".join(first_part) + "_truncate_" + str(args.max_seq_length) + "/"+second_part
        else:
           graph_dir = graph_dir + "_truncate_" + str(args.max_seq_length)
        print("graph_dir ", graph_dir)
    return graph_dir


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    data_dir = args.task_to_data_dir[task]

    if args.with_graph:
        args.graph_dir = get_graph_dir(args, task)

    if task in args.nli_task_names:
        processor = processors["nli"](data_dir, graph_dir=args.graph_dir)
    elif task.startswith("snli"):
        processor = processors["snli"](graph_dir=args.graph_dir)
    elif task in ["mnli", "mnlismall", "mnlimicro", "mnlitest"]:
        processor = processors["mnli"](graph_dir=args.graph_dir)
    elif task == "mnli-mm":
        processor = processors["mnli-mm"](graph_dir=args.graph_dir)
    elif task.startswith("HANS"):
        processor = processors["hans"](graph_dir=args.graph_dir)
    else:
        if args.binerize_train and GLUE_TASKS_NUM_LABELS[task] != 2:
            processor = processors[task](args.target_label, args.binerize_train)
        else:
            processor = processors[task]()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    print("File is: ", cached_features_file)

    if False and os.path.exists(cached_features_file) and args.use_cached_dataset:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(data_dir) if evaluate else \
            processor.get_train_examples(data_dir)

        features,max_deg = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, "classification",
                                                cls_token_at_end=bool(args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                sep_token=tokenizer.sep_token,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                with_graph=args.with_graph)

        # logger.info("Saving features into cached file %s", cached_features_file)
        # torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset 
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    

                
    if args.with_graph:

        for f in features:
            deg = len(f.graph[0])
            if deg != max_deg:
                for iii,h in enumerate(f.graph):
    
                    f.graph[iii].extend([-100]*(max_deg-deg))
                    f.graph_labels[iii].extend([-100]*(max_deg-deg))
                    if f.graph[iii][0]==-1:
                        break
        all_graph_labels = torch.tensor([f.graph_labels for f in features], dtype=torch.long)
        all_graph = torch.tensor([f.graph for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_graph_labels,
                                all_graph)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset, processor.get_labels(), processor.num_classes


def main():
    parser = argparse.ArgumentParser()
    # RUBI parameters, this is deactivated by default.
    # TODO this can be extended to having hidden states as well.
    parser.add_argument("--layernorm_key", action="store_true", help="Adds layernorm to the model.")
    parser.add_argument("--layernorm_value", action="store_true", help="Adds layernorm to the model.")
    parser.add_argument("--div_sqrt", action="store_true", help="Divides by the sqrt(dimension).")
    parser.add_argument("--batchnorm_key", action="store_true",
                        help="If specified applies batchnorm on the keys  in the relation embeddings.")
    parser.add_argument("--batchnorm_value", action="store_true",
                        help="If specified applies batchnorm on the values in the relation embeddings.")
    parser.add_argument("--do_truncate", action="store_true", help="If specified reads the graph from the\
        truncated graph paths.")
    parser.add_argument("--diff_pad_zero",default=False, action="store_true")
    parser.add_argument("--graph_dir", default=None, type=str)
    parser.add_argument("--gate_key_type", choices=[None, "key_hidden_state", "key"], default=None,
                        help="If specified, applies a gating function for the key values.")
    parser.add_argument("--gate_value_type", choices=[None, "value_hidden_states", "value"], default=None,
                        help="If specified, applies a gating function for the values.")
    parser.add_argument("--graph_expansion_type", choices=["connect_to_first_token", "connect_to_first_token_head", \
                                                           "connect_to_null"], default="connect_to_first_token_head",
                        help="Specifies the type of the extension used for creation of the graph of \
        dependency relations.")
    parser.add_argument("--binerize_sum", action="store_true", help="sum the two labels for binerization.")
    parser.add_argument("--save_labels_file", type=str, default=None, \
                        help="If specified, saves the labels.")
    parser.add_argument("--with_graph", action="store_true", help="If specified use the graph input.") #--------------------------------------------------------------------
    # Bert parameters.
    parser.add_argument("--outputfile", type=str, default=None, help="If specified, saves the results.")
    parser.add_argument("--target_label", type=str, default="entailment", help="Defines the label for the binary SNLI\
                            and join the other two labels")
    parser.add_argument("--binerize_eval", action="store_true",
                        help="If specified, it binerize the dataset. During eval")
    parser.add_argument("--binerize_train", action="store_true",
                        help="If specified, it binerize the dataset. During train")
    parser.add_argument("--use_cached_dataset", action="store_true", help="If specified will use the cached dataset")
    parser.add_argument("--model_type", default="bert", type=str, required=False, #--------------------------------------------------------------------
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="./ptmodel", type=str, required=False,#--------------------------------------------------------------------
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,#--------------------------------------------------------------------
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--eval_task_names", nargs='+', type=str, default=[], \
                        help="list of the tasks to evaluate on them.")
    parser.add_argument("--output_dir", default="./results", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train",  action='store_true',#--------------------------------------------------------------------
                        help="Whether to run training.")
    parser.add_argument("--do_eval",  action='store_true',#--------------------------------------------------------------------
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=10000,  # 50
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=10000,  # 50
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--overwrite_output_dir',  action='store_true',#-----------------------here-----------------------------------------------------------------------
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--input_labeled_graph',  action='store_true',
                        help="Input Labeled Dependency graph to the attention weights")
    parser.add_argument('--use_two_optimizer', action='store_true',
                        help="Use two optimizers for training")
    parser.add_argument("--add_classifier_to_bert_optimizer", action="store_true", help="If specified, classifier\
     parameters are also trained inside the bert optimizer.")
    parser.add_argument('--lr_nonbert', type=float, default=1e-3,
                        help="Learning rate for non-Bert params")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda")
    args.device = device

    # data_dir for all eval tasks.
    args.task_to_data_dir = {
        "mnlitest": "./preprocess/datasets/SentEval/data/GLUE/MNLItest", 
        "wnli": "./preprocess/datasets/SentEval/data/GLUE/WNLI",
        "cola": "./preprocess/datasets/SentEval/data/GLUE/CoLA",
        "qnli": "./preprocess/datasets/SentEval/data/GLUE/QNLI",
        "qqp": "./preprocess/datasets/SentEval/data/GLUE/QQP",
        "rte": "./preprocess/datasets/SentEval/data/GLUE/RTE",
        "snli": "./preprocess/datasets/SNLI",
        "snlihard": "./preprocess/datasets/SNLIHard/",
        "snlismall": "./preprocess/datasets/SentEval/data/GLUE/SNLIsmall",
        "QQP": "./preprocess/datasets/QQP/",
        "MNLIMatchedHard": "./preprocess/datasets/MNLIMatchedHard/",
        "MNLIMismatchedHard": "./preprocess/datasets/MNLIMismatchedHard/",
        "mnlimatched": "./preprocess/datasets/MNLIMatched/",
        "mnlimismatched": "./preprocess/datasets/MNLIMismatched/",
        "MNLIMatchedHardWithHardTest": "./preprocess/datasets/MNLIMatchedHardWithHardTest/",
        "MNLIMismatchedHardWithHardTest": "./preprocess/datasets/MNLIMismatchedHardWithHardTest/",
        "MNLITrueMatched": "./preprocess/datasets/MNLITrueMatched",
        "MNLITrueMismatched": "./preprocess/datasets/MNLITrueMismatched",
        "mnlismall": "./preprocess/datasets/SentEval/data/GLUE/MNLIsmall",
        "HANS-const": "./preprocess/datasets/hans/constituent",
        "HANS-lex": "./preprocess/datasets/hans/lexical_overlap",
        "HANS-sub": "./preprocess/datasets/hans/subsequence",
        "mnlimicro": "./preprocess/datasets/SentEval/data/GLUE/MNLImicro",
    }

    args.task_to_graph_dir = {

        "snli": "./preprocess/nli_parsing/semantic/slni_truncate_128",
        "snli-hard": "./preprocess/nli_parsing/semantic/slni-hard_truncate_128",
        "mnli": "./preprocess/nli_parsing/semantic/mnli",#----------------------------------------
        "mnli-mm": "./preprocess/nli_parsing/semantic/mnli",#-------------------------------

    }

    # add all variations of hans automatically
    if "HANS" in args.eval_task_names:
        # TODO: for now remove the HANS, later also parse this one.
        args.eval_task_names.remove("HANS")
        hans_variations = ["HANS-const", "HANS-lex", "HANS-sub"]
        for variation in hans_variations:
            if variation not in args.eval_task_names:
                args.eval_task_names.append(variation)

    # All of these tasks use the NliProcessor
    args.nli_task_names = ["addonerte", "dpr", "sprl", "fnplus",  "joci", "mpe", "scitail",  "sick", "sickhard", \
                           "glue", "QQP", "QQPHard", "snlihard",  "mnlimatched", "mnlimismatched", \
                           "MNLIMatchedHardWithHardTest",\
                           "MNLIMismatchedHardWithHardTest", \
                           "MNLITrueMismatched", "MNLITrueMatched", "MNLIMatchedHard", "MNLIMismatchedHard", \
                           ]
    args.actual_task_names = ["wnli", "cola", "qqp", "qnli", "rte", "snli", "sst-2", "mnli", "mnli-mm", "mrpc", "sts-b"]

    # By default we evaluate on the task itself.
    if len(args.eval_task_names) == 0:
        args.eval_task_names = [args.task_name]
    if "all" in args.eval_task_names:
        args.eval_task_names = args.nli_task_names + ["snli", "qnli", "wnli", "mnli", "qqp", "rte"]
    print(args.eval_task_names)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   -1, device, 1, bool(False), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    print("task ", args.task_name)

    if args.with_graph:
        args.graph_dir = get_graph_dir(args, args.task_name)

    if args.task_name.startswith("snli"):
        processor = processors["snli"](graph_dir=args.graph_dir)
    elif args.task_name in args.nli_task_names:
        processor = processors["nli"](args.task_to_data_dir[args.task_name])
    elif args.task_name in ["mnli-mm"]:
        processor = processors["mnli-mm"](graph_dir=args.graph_dir)
    elif args.task_name.startswith("mnli"):
        processor = processors["mnli"](graph_dir=args.graph_dir)
    elif args.task_name.startswith("HANS"):
        processor = processors["hans"](graph_dir=args.graph_dir)
    elif args.task_name in args.actual_task_names:
        if args.binerize_train and GLUE_TASKS_NUM_LABELS[args.task_name] != 2:
            processor = processors[args.task_name](args.target_label, args.binerize_train)
        else:
            processor = processors[args.task_name]()
    else:
        raise ValueError("Task not found: %s" % (args.task_name))
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, finetuning_task=args.task_name)
    config.with_graph = args.with_graph
    config.max_seq_length = args.max_seq_length
    if args.with_graph:
        config.input_labeled_graph = args.input_labeled_graph
        config.gate_key_type = args.gate_key_type
        config.gate_value_type = args.gate_value_type
        config.input_labeled_graph = args.input_labeled_graph
        config.diff_pad_zero = args.diff_pad_zero
        config.batchnorm_key = args.batchnorm_key
        config.batchnorm_value = args.batchnorm_value
        config.div_sqrt = args.div_sqrt
        config.layernorm_key = args.layernorm_key
        config.layernorm_value = args.layernorm_value

        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)

        dependency_labels = processor.get_dependency_labels()
        print(dependency_labels)
        config.dependency_labels_size = len(dependency_labels)
        config.bias_dependency_label = len(tokenizer)
        tokenizer.add_tokens(dependency_labels)
        config.unk_label_id = tokenizer.convert_tokens_to_ids('<l>:unk')

        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)
        if args.with_graph:
            # Adds dependency labels to the tokenizer and the model's embedding.
            model.resize_token_embeddings(len(tokenizer))
            model.train()
    else:
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset, _, _ = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        print("model is saved in ", os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=global_step)
            results.update(result)
            print("#################", result, "#################")

    # saves the results.
    print(results)
    if args.outputfile is not None:
        write_to_csv(results, args, args.outputfile)
    return results


if __name__ == "__main__":
    main()

