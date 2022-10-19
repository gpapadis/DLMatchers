import logging
import os
import resource
import csv
import fcntl
from datetime import datetime
import json

from config import read_arguments_train, write_config_to_file, Config
from logging_customized import setup_logging
from data_loader import load_data, DataType
from data_representation import DeepMatcherProcessor, QqpProcessor
from evaluation import Evaluation
from model import save_model
from optimizer import build_optimizer
from torch_initializer import initialize_gpu_seed
from training import train
from prediction import predict
import torch
from time import time
from sklearn.metrics import precision_score, recall_score, confusion_matrix

setup_logging()


def create_experiment_folder(model_output_dir: str, model_type: str, data_dir: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    experiment_name = "{}__{}__{}".format(data_dir.upper(), model_type.upper(), timestamp)

    output_path = os.path.join(model_output_dir, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    return experiment_name


if __name__ == "__main__":
    args = read_arguments_train()

    exp_name = create_experiment_folder(args.model_output_dir, args.model_type, args.data_dir)
    
    write_config_to_file(args, args.model_output_dir, exp_name)
    
    device, n_gpu = initialize_gpu_seed(args.seed)
    #device, n_gpu = torch.device("cpu"), 0

    if args.data_processor == "QqpProcessor":
        processor = QqpProcessor()
    else:
        # this is the default as it works for all data sets of the deepmatcher project.
        processor = DeepMatcherProcessor()

    label_list = processor.get_labels()

    logging.info("training with {} labels: {}".format(len(label_list), label_list))



    config_class, model_class, tokenizer_class = Config.MODEL_CLASSES[args.model_type]
    if config_class is not None:
       config = config_class.from_pretrained(args.model_name_or_path)
       tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
       model = model_class.from_pretrained(args.model_name_or_path, config=config)
       model.to(device)
    else:       #SBERT Models
       tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
       model = model_class.from_pretrained(args.model_name_or_path)
       model.to(device)

    logging.info("initialized {}-model".format(args.model_type))


    train_examples = processor.get_train_examples(args.data_path)


    training_data_loader = load_data(train_examples,
                                     label_list,
                                     tokenizer,
                                     args.max_seq_length,
                                     args.train_batch_size,
                                     DataType.TRAINING, args.model_type)
    logging.info("loaded {} training examples".format(len(train_examples)))

    num_train_steps = len(training_data_loader) * args.num_epochs

    optimizer, scheduler = build_optimizer(model,
                                           num_train_steps,
                                           args.learning_rate,
                                           args.adam_eps,
                                           args.warmup_steps,
                                           args.weight_decay)
    logging.info("Built optimizer: {}".format(optimizer))

    eval_examples = processor.get_dev_examples(args.data_path)
    evaluation_data_loader = load_data(eval_examples,
                                       label_list,
                                       tokenizer,
                                       args.max_seq_length,
                                       args.eval_batch_size,
                                       DataType.EVALUATION, args.model_type)

    evaluation = Evaluation(evaluation_data_loader, exp_name, args.model_output_dir, len(label_list), args.model_type)
    logging.info("loaded and initialized evaluation examples {}".format(len(eval_examples)))

    t1 = time()
    train(device,
          training_data_loader,
          model,
          optimizer,
          scheduler,
          evaluation,
          args.num_epochs,
          args.max_grad_norm,
          args.save_model_after_epoch,
          experiment_name=exp_name,
          output_dir=args.model_output_dir,
          model_type=args.model_type)
    t2 = time()
    training_time = t2-t1
    train_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    #Testing
    test_examples = processor.get_test_examples(args.data_path)

    logging.info("loaded {} test examples".format(len(test_examples)))
    test_data_loader = load_data(eval_examples,
                                 label_list,
                                 tokenizer,
                                 args.max_seq_length,
                                 args.eval_batch_size,
                                 DataType.TEST, args.model_type)

    include_token_type_ids = False
    if args.model_type == 'bert':
       include_token_type_ids = True
       
    t1 = time()
    simple_accuracy, f1, classification_report, prfs, predictions = predict(model, device, test_data_loader, include_token_type_ids)
    t2 = time()
    testing_time = t2-t1
    test_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    logging.info("Prediction done for {} examples.F1: {}, Simple Accuracy: {}".format(len(test_data_loader), f1, simple_accuracy))

    logging.info(classification_report)

    #logging.info(predictions)         
    
    keys = ['precision', 'recall', 'fbeta_score', 'support']
    prfs = {f'class_{no}': {key: float(prfs[nok][no]) for nok, key in enumerate(keys)} for no in range(2)}
    
    with open('test_scores.txt', 'a') as fout:
        scores = {'simple_accuracy': simple_accuracy, 'f1': f1, 'model_type': args.model_type,
         'data_dir': args.data_dir, 'training_time': training_time, 'testing_time': testing_time, 'prfs': prfs}
        fout.write(json.dumps(scores)+"\n")

 
    '''     
    save_model(model, exp_name, args.model_output_dir, tokenizer=tokenizer)
    '''

    # Generate stats
    predicted_class = predictions['predictions']
    labels = predictions['labels']
    p = precision_score(y_true=labels, y_pred=predicted_class)
    r = recall_score(y_true=labels, y_pred=predicted_class)
    f_star = 0 if (p + r - p * r) == 0 else p * r / (p + r - p * r)
    tn, fp, fn, tp = confusion_matrix(y_true=labels,
                                      y_pred=predicted_class).ravel()

    # Persist Results
    result_file = '/home/remote/u6852937/projects/results.csv'
    file_exists = os.path.isfile(result_file)

    with open(result_file, 'a') as results_file:
      heading_list = ['method', 'dataset_name', 'train_time',
                      'test_time',
                      'train_max_mem', 'test_max_mem', 'TP', 'FP',
                      'FN',
                      'TN', 'Pre', 'Re', 'F1', 'Fstar']
      writer = csv.DictWriter(results_file, fieldnames=heading_list)

      if not file_exists:
        writer.writeheader()

      fcntl.flock(results_file, fcntl.LOCK_EX)
      result_dict = {
        'method': 'emtransformer-{}-epochs-{}'.format(args.model_type,args.num_epochs),
        'dataset_name': args.data_dir,
        'train_time': round(training_time, 2),
        'test_time': round(testing_time, 2),
        'train_max_mem': train_max_mem,
        'test_max_mem': test_max_mem,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Pre': round(p * 100, 2),
        'Re': round(r * 100, 2),
        'F1':  round(f1 * 100, 2),
        'Fstar': round(f_star * 100, 2),
      }
      writer.writerow(result_dict)
      fcntl.flock(results_file, fcntl.LOCK_UN)
