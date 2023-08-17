from argparse import ArgumentParser, ArgumentTypeError
from datetime import datetime
import os
import yaml
import numpy as np
import torch
import random
from data.get_datasets import get_class_splits
import logging


def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s', level=getattr(logging, level))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def over_write_args_from_file(args, yml):
    """
    overwrite arguments acocrding to config file
    """
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--c", default="config.yml", type=str, help="config file to use")

    parser.add_argument("--checkpoint_dir", default="checkpoints", type=str, help="checkpoint dir")
    parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
    parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
    parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")

    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--alpha", default=0.1, type=float, help="weight of kd loss")

    parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
    parser.add_argument('--amp', default=False, type=str2bool, help='use mixed precision training or not')
    parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
    parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
    parser.add_argument("--max_epochs", default=10, type=int, help="warmup epochs")

    parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
    parser.add_argument("--embedding_dim", default=512, type=int, help="projected dim")
    parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
    parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
    parser.add_argument("--num_heads", default=1, type=int, help="number of heads for clustering")
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
    parser.add_argument("--project", default="KT", type=str, help="wandb project")
    parser.add_argument("--entity", default="kleinzcy", type=str, help="wandb entity")
    parser.add_argument("--offline", default=True, type=str2bool, help="disable wandb")
    parser.add_argument("--eval", default=False, type=str2bool, help="train or eval")

    parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
    parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
    parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
    parser.add_argument("--multicrop", default=False, type=str2bool, help="activates multicrop")
    parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
    parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")
    parser.add_argument("--use_ssb_splits", default=False, type=str2bool, help="use ssb splits")

    parser.add_argument("--gpus", type=str, default="0", help="the gpu")
    parser.add_argument("--resume", default=False, type=str2bool, help="whether to use old model")
    parser.add_argument("--save-model", default=False, type=str2bool, help="whether to save model")
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--eval_func', default="v3", type=str)
    parser.add_argument('--unknown_cluster', default=False, type=str2bool)
    parser.add_argument("--cluster_error_rate", default=0, type=float, help="softmax temperature")
    parser.add_argument("--kd_temperature", default=4, type=float, help="softmax temperature")
    parser.add_argument("--prop_train_labels", default=0.5, type=float, help="prop train samples")

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    os.environ["WANDB_API_KEY"] = "Your Key"
    os.environ["WANDB_MODE"] = "offline" if args.offline else "online"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    args.model_save_dir = os.path.join(args.checkpoint_dir, "pretrain" if args.pretrain else "discover", args.alg, args.dataset, args.comment)
    args.log_dir = args.model_save_dir

    if not args.eval:
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        elif args.resume or "debug" in args.comment or "repeat" in args.comment or "analysis" in args.comment or args.eval:
            print(f"Resume is {args.resume}, comment is :{args.comment}")
        else:
            pass
            # raise FileExistsError("Duplicate exp name {}. Please rename the exp name!".format(args.model_save_dir))
    args.low_res = "cifar" in args.dataset.lower()
    # TODO: design a special select function to select classes

    seed = 2023
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    # args.train_classes = np.random.choice(args.num_classes, args.num_unlabeled_classes, replace=False)
    args = get_class_splits(args)

    print(args)
    return args