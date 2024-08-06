import argparse
import copy
import torch
import os
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from dassl.data import DataManager
# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.brightness
import datasets.contrast
import datasets.defocus
import datasets.elastic
import datasets.fog
import datasets.frost
import datasets.gaussian
import datasets.glass
import datasets.impluse
import datasets.jpeg
import datasets.motion
import datasets.pixelate
import datasets.shot
import datasets.snow
import datasets.zoom

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.imagenet_c
import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.independentVL
import trainers.scp
import time


import smtplib
from email.message import EmailMessage

def send_email(subject, body, to_email=None):
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = 'ikuta143@gmail.com'
    msg['To'] = 'rangowang6@gmail.com'
    
    # Establish a connection to Gmail
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    
    # Login to your Gmail account
    server.login('ikuta143@gmail.com', 'lieber143.')
    
    # Send the email
    server.send_message(msg)
    server.quit()
def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for SCP
    cfg.TRAINER.SCP = CN()
    cfg.TRAINER.SCP.N_CTX_VISION = 4  # number of context vectors at the vision branch
    cfg.TRAINER.SCP.N_CTX_TEXT = 4  # number of context vectors at the language branch
    cfg.TRAINER.SCP.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.SCP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.SCP.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)
    cfg.TRAINER.SCP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will be using shallow IVLP prompting (J=1)


    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow IVLP prompting(J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    #continual testing all datasets
    all_dataset_names = []

    # 1. From the dataset config files
    if args.dataset_config_file:
        dataset_config_files = args.dataset_config_file.split(',')
        for dataset_config_file in dataset_config_files:
            dataset_config_file = f'configs/datasets/{dataset_config_file.strip()}.yaml'
            if os.path.exists(dataset_config_file):
                temp_cfg = cfg.clone()  
                temp_cfg.merge_from_file(dataset_config_file)  
                all_dataset_names.append(temp_cfg.DATASET.NAME)  
            else:
                print(f"Warning: Config file {dataset_config_file} does not exist!")

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    # 5. Print config
    cfg.DATASET.NAME = ','.join(all_dataset_names)

    return cfg





def main(args):
    start_time = time.time()
    if args.eval_only:
        cfg_master = setup_cfg(args)
        dataset_names = cfg_master.DATASET.NAME.split(',')
        print("The dataset names: ", dataset_names)
        start_time = time.time()
        trainer = None  
        for dataset_name in dataset_names:
            cfg = copy.deepcopy(cfg_master)
            cfg.defrost() 
            cfg.DATASET.NAME = dataset_name.strip()
            cfg.freeze()  

            if cfg.SEED >= 0:
                print("Setting fixed seed: {}".format(cfg.SEED))
                set_random_seed(cfg.SEED)

            setup_logger(cfg.OUTPUT_DIR)

            trainer = build_trainer(cfg)

            print("----------------------------------------------------")
            print("Loading dataset:", dataset_name)
            trainer.test()
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total training time: {total_time:.2f} seconds")
        return




    if not args.no_train:
        trainer.train()
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="SCP", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
