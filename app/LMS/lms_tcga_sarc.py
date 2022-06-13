"""
Author       : Asmaa Aljuhani
Description  : Weakly supervised UACNN application for LMS FNCLCC grade classification
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from config import TCGA_SARC_Config,  TCGA_SARC_InferenceConfig


#Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
from uacnn.uacnn import UACNN

class SARC():
    def prepare_dataset(self, dataset_dir, labels_df, config):
        #originl dataframe
        sarc_df = pd.read_csv(labels_df)
        print(f'There are {sarc_df.shape[0]} rows and {sarc_df.shape[1]} columns')

        labels = []
        unique_var = len(set(sarc_df[config.TASK]))
        labels.append({
            "name": config.TASK,
            "nclasses": unique_var
        })
        print("labels:", json.dumps(labels, indent=4))
        return sarc_df, labels

    def train(self, mode, config, dataset_dir, train_data_desc, val_data_desc, log_dir , weights):
        print("Training")
        # init model
        uacnn = UACNN(mode=mode,
                      cfg=config,
                      log_dir=log_dir,
                      weights=weights)

        # load data
        train_data_df, train_labels = self.prepare_dataset(dataset_dir,train_data_desc, config=config)
        val_data_df, val_labels = self.prepare_dataset(dataset_dir, val_data_desc, config=config)

        uacnn.load_dataset(dataset_dir=dataset_dir,
                           train_data_df= train_data_df, train_labels=train_labels,
                           val_data_df=val_data_df, val_labels=train_labels)

        # build model
        uacnn.model(labels=train_labels)

        # train model
        uacnn.train()

    def eval_uncertainty(self, mode, config,dataset_dir, train_data_desc, val_data_desc, log_dir , weights):
        print("Eval uncertainty")
        # init model #TODO: load model
        uacnn = UACNN(mode=mode,
                                cfg=config,
                                log_dir=log_dir,
                                weights=weights)

        # load data
        train_data_df, train_labels = self.prepare_dataset(dataset_dir,train_data_desc, config=config)
        val_data_df, val_labels = self.prepare_dataset(dataset_dir, val_data_desc, config=config)

        uacnn.load_dataset(dataset_dir=dataset_dir,
                                train_data_df= train_data_df, train_labels=train_labels,
                                val_data_df=val_data_df, val_labels=train_labels)

        # build model
        uacnn.model(labels=train_labels)

        # eval
        uacnn.eval_uncertainty()


    def wsi_uncertainty(self, mode, config, wsi_dir, log_dir , weights, tile_size, mag, task, nclasses, class2index, tempscaler=None):
        print("WSI detection")
        uacnn = UACNN(mode=mode,
                                cfg=config,
                                log_dir=log_dir,
                                weights=weights)
        # build model
        labels = [{"name": task,"nclasses": int(nclasses)}]

        # build model
        uacnn.model(labels=labels)


        # used to evaluate validation datset and compute temp scaler
        uacnn.wsi_uncertainty_map(wsi_dir, tile_size, mag, nclasses, class2index)


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Uncertainty Aware CNN')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'eval' or 'eval_uncertainty' or 'wsi_uncertainty_map'")
    parser.add_argument('--img_dir', required=False,
                        metavar="/path/to/dataset/")
    parser.add_argument('--data_desc', required=False,
                        metavar="/path/to/data_desc_file/")
    parser.add_argument('--train_data_desc', required=False,
                        metavar="/path/to/data_desc_file/")
    parser.add_argument('--val_data_desc', required=False,
                        metavar="/path/to/data_desc_file/")
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'imagenet' or last")
    parser.add_argument('--logs', required=False,
                        # default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    #the following args are for wsi inferencing
    parser.add_argument('--tile_size', required=False,
                        metavar="256 or 512",
                        help='Provide tile size to detect')
    parser.add_argument('--mag', required=False,
                        metavar="5, 10, 20",
                        help='Provide magnification for detection')
    parser.add_argument('--task', required=False,
                        help='Provide class/label to be trained on')
    parser.add_argument('--nclasses', required=False,
                        help='Provide number classes to be trained on')
    parser.add_argument('--class2index', required=False,
                        help="Provide an array of class names (i.e 1,2,3)")
    parser.add_argument('--tempscaler', required=False,
                        help="Provide tempscaler")

    args = parser.parse_args()


    # Configurations
    if args.command == "train":
        config = TCGA_SARC_Config()
    else:
        config = TCGA_SARC_InferenceConfig()

    config.FOLDER = os.path.basename(os.path.dirname(args.img_dir))
    config.IMG_DIR = args.img_dir
    config.WEIGHTS = args.weights
    config.TRAIN_DATA_DESC = args.train_data_desc
    config.VAL_DATA_DESC = args.val_data_desc
    config.TASK= args.task
    config.NCLASSES = args.nclasses
    if(args.class2index):
        config.class2index =  str.split(args.class2index,',')
    if(args.tempscaler):
        config.TEMPSCALER = args.tempscaler
    config.display()

    # Initiate the application class
    sarc = SARC()

    # Create model
    if args.command == "train":
        print("*************************** Training **************************")
        sarc.train(mode="training",
                   config=config,
                   dataset_dir=args.img_dir,
                   train_data_desc = args.train_data_desc,
                   val_data_desc = args.val_data_desc,
                   log_dir=args.logs,
                   weights=args.weights)
    elif args.command == "eval_uncertainty":
        sarc.eval_uncertainty(mode="inference",
                        config=config,
                        dataset_dir=args.img_dir,
                        train_data_desc = args.train_data_desc,
                        val_data_desc=args.val_data_desc,
                        log_dir=args.logs,
                        weights=args.weights)
    elif args.command == "wsi_uncertainty_map":
        sarc.wsi_uncertainty(mode="inference",
                                  config=config,
                                  wsi_dir=args.img_dir,
                                  log_dir=args.logs,
                                  weights=args.weights,
                                  tile_size=args.tile_size,
                                  mag=args.mag,
                                  task=args.task,
                                  nclasses = args.nclasses,
                                  class2index= config.class2index)

