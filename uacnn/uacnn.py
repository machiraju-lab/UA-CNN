"""
Author       : Asmaa Aljuhani
Date         : 03/23/2020
Description  : UACNN is a Resnet CNN implementation that utilize transfer learning and Monte Carlo dropouts for uncertainty quantification
"""
import csv
import json
import os
import re
import datetime
import glob
import large_image
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tqdm import tqdm
import itertools
import histomicstk as htk
import torch
from torchvision import transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from shapely.geometry import Polygon
from PIL import Image, ImageColor
from preprocessing.tissue_detection import DetectTissue
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, images_folder, labels, class2index, transform=None):
        self.df = df
        self.images_folder = images_folder
        self.transform = transform
        self.labels = labels
        self.class2index = class2index
        self.classes = np.ravel(self.df[[self.labels[0]['name']]])
        self.class_weights = torch.tensor(len(self.classes) / (len(np.unique(self.classes)) * np.bincount(self.classes).astype(np.float64))[1:]).float()
        self.class_weights_dict = dict(((el, w.item()) for el, w in zip(np.unique(self.classes), self.class_weights)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df["tiles"][index]
        label = torch.tensor(float(self.class2index.index(self.df[self.labels[0]['name']].astype(str)[index])))
        try:
            image = Image.open(os.path.join(self.images_folder, filename)).convert('RGB')
        except:
            print('no imge:', filename)
            return None
        if self.transform is not None:
            image = self.transform(image)
        return image, label, filename


class UACNN():
    """
    Uncertainty Aware convolutional neural networks (UA-CNN)
    """

    def __init__(self, mode, cfg, log_dir, weights):
        """
        :param mode: "trainining or "inference"
        :param config:  subcalss of the Config class
        :param log_dir: Directory to save training logs and trained weights
        :param weights: Options: "imagenet", "last", "path", "histossl"
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.cfg = cfg
        if mode == "training":
            self.set_log_dir(log_dir)
            self.writer = SummaryWriter(log_dir=self.log_dir)
        self.set_weigths(weights)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print("device", self.device)


    # https://github.com/pytorch/pytorch/issues/1137
    def collate_fn(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def load_dataset(self, dataset_dir, train_data_df=None, train_labels=None, val_data_df=None, val_labels=None):
        """

        :param dataset_dir:
        :param train_data_df:
        :param train_labels:
        :param val_data_df:
        :param val_labels:
        :return:
        """
        print("Load_dataset")

        if self.mode == "training":

            data_transforms = {
                'train':
                    transforms.Compose([
                        transforms.Resize((self.cfg.IMAGE_RESIZE, self.cfg.IMAGE_RESIZE)),
                        transforms.RandomHorizontalFlip(p=.25),
                        transforms.RandomVerticalFlip(p=.25),
                        self.deconvolution_normalization(),
                        transforms.ToTensor(),
                    ]),
                'val':
                    transforms.Compose([
                        transforms.Resize((self.cfg.IMAGE_RESIZE, self.cfg.IMAGE_RESIZE)),
                        self.deconvolution_normalization(),
                        transforms.ToTensor(),
                    ]),
                'test':
                    transforms.Compose([
                        transforms.Resize((self.cfg.IMAGE_RESIZE, self.cfg.IMAGE_RESIZE)),
                        self.deconvolution_normalization(),
                        transforms.ToTensor(),
                    ])
            }

            # get classes from training data
            # This assume that training data have all classes
            self.class2index = sorted(train_data_df[train_labels[0]['name']].astype(str).unique())
            print('classes', self.class2index)

            train_dataset = MyDataset(train_data_df, dataset_dir, train_labels, self.class2index,data_transforms['train'])
            val_dataset = MyDataset(val_data_df, dataset_dir, val_labels, self.class2index, data_transforms['val'])
            test_dataset = MyDataset(val_data_df, dataset_dir, val_labels, self.class2index, data_transforms['test'])

            self.dataloaders = {
                'train': torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=self.cfg.BATCH_SIZE_TRAINING,
                                                     shuffle=True,
                                                     collate_fn=self.collate_fn,
                                                     num_workers=self.cfg.NUM_WORKERS,  # for Kaggle
                                                     ),
                'val': torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=self.cfg.BATCH_SIZE_VALIDATION,
                                                   shuffle=True,
                                                   collate_fn=self.collate_fn,
                                                   num_workers=self.cfg.NUM_WORKERS,  # for Kaggle
                                                   ),
                'test': torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    collate_fn=self.collate_fn,
                                                    num_workers=self.cfg.NUM_WORKERS
                                                    )
            }

            print(f'Num training images: {len(self.dataloaders["train"].dataset)}')
            print(f'Num validation images: {len(self.dataloaders["val"].dataset)}')

        else:
            print('Testing')
            data_transforms = {
                'train':
                    transforms.Compose([
                        transforms.Resize((self.cfg.IMAGE_RESIZE, self.cfg.IMAGE_RESIZE)),
                        self.deconvolution_normalization(),
                        transforms.ToTensor(),
                    ]),
                'test':
                    transforms.Compose([
                        transforms.Resize((self.cfg.IMAGE_RESIZE, self.cfg.IMAGE_RESIZE)),
                        self.deconvolution_normalization(),
                        transforms.ToTensor(),
                    ])
            }

            # get classes from training data
            # This assume that training data have all classes
            self.class2index = sorted(train_data_df[train_labels[0]['name']].astype(str).unique())
            print('classes', self.class2index)
            # df, images_folder, labels, transform = None):
            train_dataset = MyDataset(train_data_df, dataset_dir, train_labels, self.class2index,data_transforms['train'])
            test_dataset = MyDataset(val_data_df, dataset_dir, val_labels, self.class2index, data_transforms['test'])

            self.dataloaders = {
                'train': torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=self.cfg.BATCH_SIZE_TRAINING,
                                                     shuffle=True,
                                                     collate_fn=self.collate_fn,
                                                     num_workers=0),  # for Kaggle
                'test': torch.utils.data.DataLoader(test_dataset,
                                                    batch_size=self.cfg.BATCH_SIZE_TESTING,
                                                    shuffle=False,
                                                    collate_fn=self.collate_fn,
                                                    num_workers=0)  # for Kaggle
            }

    def model(self, labels):
        """
        Build UACNN architicture
        :param labels:
        :return:
        """
        assert self.mode in ['training', 'inference']
        # https://wtfleming.github.io/2020/04/12/pytorch-cats-vs-dogs-part-3/

        if self.mode == 'training':
            if self.cfg.WEIGHTS == "imagenet":
                # Download models pretrained on Imagenet
                self.model_resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

            elif self.cfg.WEIGHTS == "histossl":
                # source: https://github.com/ozanciga/self-supervised-histopathology
                self.model_resnet18 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
                state = torch.load("../../uacnn/tenpercent_resnet18.ckpt", map_location=self.device)
                state_dict = state['state_dict']
                for key in list(state_dict.keys()):
                    state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

                # load model weights
                model_dict = self.model_resnet18.state_dict()
                weights = {k: v for k, v in state_dict.items() if k in model_dict}
                if weights == {}:
                    print('No weight could be loaded..')
                model_dict.update(weights)
                self.model_resnet18.load_state_dict(model_dict)

            # freeze all params except the BatchNorm layers
            for name, param in self.model_resnet18.named_parameters():
                if ("bn" not in name):
                    param.requires_grad = False

            self.model_resnet18.fc = nn.Sequential(
                nn.Linear(self.model_resnet18.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(.5),
                nn.Linear(128, labels[0]['nclasses']))

            self.model_resnet18.to(self.device)
            self.optimizer = optim.Adam(self.model_resnet18.parameters(), lr=0.001)
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=self.dataloaders["train"].dataset.class_weights.to(self.device))
            # Decay LR by a factor of 0.1 every 7 epochs
            self.exp_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
            print('model', self.model_resnet18)

        elif self.mode == 'inference':
            print('inference')
            # used for transforming wsi tiles
            self.tile_transforms = transforms.Compose([
                transforms.Resize((self.cfg.IMAGE_RESIZE, self.cfg.IMAGE_RESIZE)),
                self.deconvolution_normalization(),
                transforms.ToTensor()
            ])

            self.model_resnet18 = torch.hub.load('pytorch/vision', 'resnet18')
            # freeze all params except the BatchNorm layers
            for name, param in self.model_resnet18.named_parameters():
                if ("bn" not in name):
                    param.requires_grad = False
            self.model_resnet18.fc = nn.Sequential(
                nn.Linear(self.model_resnet18.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(.5),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(.5),
                nn.Linear(128, labels[0]['nclasses']))

            self.model_resnet18.load_state_dict(torch.load(self.model_weights, map_location=self.device))
            self.model_resnet18.eval()
            print(self.model_resnet18)

    def get_histossl_weights(self):
        """
        source: https://github.com/ozanciga/self-supervised-histopathology
        Laod HistoSSL weights
        :return:
        """
        MODEL_PATH = 'tenpercent_resnet18.ckpt'
        RETURN_PREACTIVATION = False  # return features from the model, if false return classification logits
        NUM_CLASSES = 4  # only used if RETURN_PREACTIVATION = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        print('model_path', model_path)

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            weights = glob.glob(model_path + "/*.h5")
            self.last_weights = max(weights, key=os.path.getctime)
            print('last_weight', self.last_weights)
            regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/uacnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, self.last_weights)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6))
                print('Re-starting from epoch {}'.format(self.epoch + 1))
                # Directory for training logs
                self.log_dir = os.path.join(model_path)  # , "{}{:%Y%m%dT%H%M}".format(self.cfg.NAME.lower(), now))

        else:
            # Directory for training logs
            self.log_dir = os.path.join(self.cfg.DEFAULT_LOGS_DIR,
                                        "{}/{}_{}_{:%Y%m%dT%H%M}".format(self.cfg.FOLDER, self.cfg.NAME.lower(),
                                                                         self.cfg.TASK, now))
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

        # Create log_dir if not exists
        if not os.path.exists(self.cfg.DEFAULT_LOGS_DIR):
            os.makedirs(self.cfg.DEFAULT_LOGS_DIR)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "uacnn_{}_*epoch*.h5".format(self.cfg.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

        print('log_dir', self.log_dir)
        print('checkpoint_path', self.checkpoint_path)

    def set_weigths(self, weights_path):
        if weights_path == "imagenet":
            self.model_weights = None
        elif weights_path == "last":
            self.model_weights = self.last_weights
        else:
            self.model_weights = weights_path

    def plot_confusion_matrix(self, cm, class_names, tensor_name='MyFigure/image'):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float'), decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig("{}/{}_CM.png".format(os.path.dirname(self.model_weights), self.cfg.NAME))

    def plot_roc(self, test_labels, test_pred, tensor_name='MyFigure/image'):

        test_fp, test_tp, _ = sklearn.metrics.roc_curve(test_labels, test_pred)
        figure = plt.figure(figsize=(8, 8))

        plt.plot(100 * test_fp, 100 * test_tp, label="Test Baseline", linewidth=2)

        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        # plt.xlim([80,100.5])
        # plt.ylim([80,100.5])
        plt.grid(True)
        plt.legend(loc='lower right')
        ax = plt.gca()
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig("{}/{}_ROC.png".format(os.path.dirname(self.model_weights), self.cfg.NAME))

    def deconvolution_normalization(self):
        def normalization(input_image):
            # TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
            # for macenco (obtained using rgb_separate_stains_macenko_pca()
            # and reordered such that columns are the order:
            # Hamtoxylin, Eosin, Null
            W_target = np.array([
                [0.5807549, 0.08314027, 0.08213795],
                [0.71681094, 0.90081588, 0.41999816],
                [0.38588316, 0.42616716, -0.90380025]
            ])
            try:
                norm = htk.preprocessing.color_normalization.deconvolution_based_normalization(np.array(input_image),
                                                                                               W_target=W_target)
            except Exception as e:
                norm = input_image
            return norm
        return normalization


    def train(self):
        print('train hsitomtcnn')
        # remember best acc and save checkpoint
        is_best = False
        best_acc = 0
        min_loss = float('inf')

        for epoch in tqdm(range(self.cfg.EPOCHS)):
            training_loss = 0.0
            valid_loss = 0.0

            self.model_resnet18.train()
            num_train_correct = 0
            num_train_examples = 0

            for batch in tqdm(self.dataloaders['train']):
                self.optimizer.zero_grad()
                inputs, targets, fnames = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device, dtype=torch.long)
                output = F.log_softmax(self.model_resnet18(inputs))
                loss = self.loss_fn(output, targets)
                loss.backward()
                self.optimizer.step()
                training_loss += loss.data.item() * inputs.size(0)
                pred = torch.max(output, dim=1)[1]
                is_correct = torch.eq(pred, targets).view(-1)
                num_train_correct += torch.sum(is_correct).item()
                num_train_examples += pred.shape[0]
            training_loss /= len(self.dataloaders['train'].dataset)
            print('done train batch')
            self.model_resnet18.eval()
            print('done eval()')
            num_val_correct = 0
            num_val_examples = 0
            for batch in tqdm(self.dataloaders['val']):
                inputs, targets, fnames = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device, dtype=torch.long)
                output = F.log_softmax(self.model_resnet18(inputs))
                loss = self.loss_fn(output, targets)
                valid_loss += loss.data.item() * inputs.size(0)
                pred = torch.max(output, dim=1)[1]
                is_correct = torch.eq(pred, targets).view(-1)
                num_val_correct += torch.sum(is_correct).item()
                num_val_examples += pred.shape[0]
            valid_loss /= len(self.dataloaders['val'].dataset)

            # write to tensorboard
            train_acc = num_train_correct / num_train_examples
            val_acc = num_val_correct / num_val_examples
            is_best = (valid_loss < min_loss) or (val_acc > best_acc)
            best_acc = max(val_acc, best_acc)
            min_loss = min(valid_loss, min_loss)

            self.writer.add_scalar("Loss/train", training_loss, epoch)
            self.writer.add_scalar("Loss/val", valid_loss, epoch)
            self.writer.add_scalar("acc/train", train_acc, epoch)
            self.writer.add_scalar("acc/val", val_acc, epoch)

            print(
                'Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, train_accuracy = {:.4f}, val_accuracy = {:.4f}'.format(
                    epoch,
                    training_loss,
                    valid_loss,
                    train_acc,
                    val_acc))

            # save weights for min val loss or high val accuracy
            if is_best:
                torch.save(self.model_resnet18.state_dict(),
                           os.path.join(self.log_dir, "model_resnet18_{}.pth".format(epoch)))
        self.writer.close()

    def enable_dropout(self, m):
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

    def h_entropy(self, p):
        return -1 * np.mean(p * np.log2(p), axis=0)


    def eval_uncertainty(self):
        print("eval_uncertainty")
        # predict stochastic dropout model T times
        T = 10
        correct = 0
        total = 0
        pred = []
        gt = []
        file_names = []

        # enable dropout
        self.enable_dropout(self.model_resnet18)

        f = open("{}_eval_uncertainty_prediction.csv".format(os.path.dirname(self.model_weights), self.cfg.NAME), "w")
        f.write("image,gt,pred_class,pred_prob,classes_probs,,,h_entropy,,,\n")
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['test']):
                images, labels, samples_fname = batch
                images = images.to(self.device)
                labels = labels.to(self.device, dtype=torch.long)

                p_hat = []
                for t in range(T):
                    outputs = self.model_resnet18(images)
                    outputs = F.log_softmax(outputs, dim=1)
                    prob = torch.exp(outputs)
                    p_hat.append(prob.cpu().data.numpy()[0])
                # convert to numpy array
                p_hat = np.array(p_hat)

                #  compute mean prediction
                prediction = np.mean(p_hat, axis=0)  # prob of all classes
                predicted = np.argmax(prediction)  # index of maximum prediction
                max_prob = np.max(prediction)  # prob of maximum prediction
                # estimate uncertainties
                h_entopy = self.h_entropy(p_hat)

                file_names.append(samples_fname)
                gt.append(labels.item())
                pred.append(predicted)

                f.write(
                    "{}, {}, {}, {}, {}, {}\n".format(samples_fname[0],
                                                                  labels.item(),
                                                                  predicted,
                                                                  max_prob,
                                                                  prediction.tolist(),
                                                                  h_entopy.tolist()
                                                                  ))

                total += labels.size(0)
                correct += (predicted.astype(float) == labels).sum()

        f.close()
        print('correct: {:d}  total: {:d}'.format(correct, total))
        print('accuracy = {:f}'.format(correct / total))

        print(classification_report(gt, pred, target_names=list(map(str, self.class2index))))
        # Calculate the confusion matrix.
        cm = confusion_matrix(gt, pred)  # , normalize="all")
        print('cm', cm)
        self.plot_confusion_matrix(cm, self.class2index)

    def wsi_uncertainty_map(self, wsi_dir, tile_size, mag, nclasses, class2index, calibration_method=None, **kwargs):
        print("detect_wsi", wsi_dir)
        print('mag', mag)
        print('tile', tile_size)
        print('nclasses', nclasses)
        print('class2index', class2index)
        print('Variational dropout: ', 10)

        # Detect tissue region
        detect_tissue = DetectTissue()
        tissue_regions = detect_tissue.get_wsi_mask(wsi_dir)
        plt.clf()
        # # utils.to_qupath_ann(tissue_regions, wsi_dir)

        # TCGA-A2-A3XS-DX1_xmin21421_ymin37486_.png, Amgad et al, 2019)
        # for macenco (obtained using rgb_separate_stains_macenko_pca()
        # and reordered such that columns are the order:
        # Hamtoxylin, Eosin, Null
        W_target = np.array([
            [0.5807549, 0.08314027, 0.08213795],
            [0.71681094, 0.90081588, 0.41999816],
            [0.38588316, 0.42616716, -0.90380025]
        ])

        # Load wsi
        ts = large_image.getTileSource(wsi_dir)
        predection_layer = dict()
        i = 0
        T = 10
        # enable dropout
        self.enable_dropout(self.model_resnet18)
        predicted_tiles = np.zeros(int(nclasses))
        total_tiles = 0
        color = self.cfg.TASKS_COLORS[self.cfg.TASKS.index(self.cfg.TASK)]

        for tile_info in tqdm(ts.tileIterator(
                scale=dict(magnification=mag),
                tile_size=dict(width=tile_size, height=tile_size),
                # tile_overlap=dict(x=tile_overlap_x, y=tile_overlap_y),
                format=large_image.tilesource.TILE_FORMAT_PIL)):
            # img
            im_tile = tile_info['tile']
            tile_polygon = Polygon([(tile_info['gx'], tile_info['gy']),
                                    (tile_info['gx'] + tile_info['gwidth'], tile_info['gy']),
                                    (tile_info['gx'] + tile_info['gwidth'], tile_info['gy'] + tile_info['gheight']),
                                    (tile_info['gx'], tile_info['gy'] + tile_info['gheight'])])

            '''
            # convert to HSV-space, then split into channels
            hsv = cv2.cvtColor(im_tile, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            # print(s)
            # threshold the S channel using adaptive method(`THRESH_OTSU`)
            th, threshed = cv2.threshold(s, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            if th < 45:
                pass
            '''

            for tissue_region in tissue_regions:
                # check if a tile has a tissue
                # tissue_intersection = tissue_region.intersection(tile_polygon)
                # if (tissue_region['polygon'].contains(tile_polygon)): # good for wsi_tiling
                tissue_intersection_area = tissue_region['polygon'].intersection(tile_polygon).area
                # print('area', tissue_intersection_area)

                if (tissue_intersection_area > ((int(tile_size) ^ 2) / 2)):  # good for prediction

                    # im_tile.save('test/img_tile_{}.png'.format(i))
                    # i = i + 1
                    ## PREDECTION
                    img = self.tile_transforms(im_tile.convert("RGB"))
                    img = img.unsqueeze(0)
                    # predict multiple times
                    p_hat = []
                    for t in range(T):
                        outputs = self.model_resnet18(img)
                        if calibration_method:
                            outputs = calibration_method(outputs, kwargs)
                        outputs = F.log_softmax(outputs, dim=1)
                        # prob = F.log_softmax(outputs, dim=1)
                        prob = torch.exp(outputs)
                        p_hat.append(prob.data.numpy()[0])
                    # convert to numpy array
                    p_hat = np.array(p_hat)
                    # p_hat = np.exp(np.array(p_hat)) used with log_softmax
                    #  compute mean prediction
                    prediction = np.mean(p_hat, axis=0)  # prob of all classes
                    predicted = np.argmax(prediction)  # index of maximum prediction
                    max_prob = np.max(prediction)  # prob of maximum prediction
                    # print('arg max', np.max(prediction))
                    # print('arg argmax', np.argmax(prediction)) # index of maximum prediction
                    # estimate uncertainties
                    aleatoric, epistemic = self.uncertainties(p_hat)
                    uncertainties = aleatoric + epistemic
                    h_entopy = self.h_entropy(p_hat)
                    pred_class = predicted.item()
                    # only add predicted tiles if maxprob >  90%
                    if (max_prob > .80 and h_entopy[pred_class] < 0.2):
                        predicted_tiles[pred_class] = predicted_tiles[pred_class] + 1
                        if pred_class in predection_layer:
                            predection_layer[pred_class].append(
                                {'name': pred_class,
                                 'color': self.cfg.TASKS_COLORS[self.cfg.TASKS.index(self.cfg.TASK) - pred_class],
                                 'conf': max_prob,
                                 'polygon': tile_polygon})
                        else:
                            predection_layer[pred_class] = [
                                {'name': pred_class,
                                 'color': self.cfg.TASKS_COLORS[self.cfg.TASKS.index(self.cfg.TASK) - pred_class],
                                 'conf': max_prob,
                                 'polygon': tile_polygon}]
                    total_tiles = total_tiles + 1

        self.to_histomicstk_heatmap_ann(wsi_dir, predection_layer, "calibrated_confidence_clean80_FNCLCC")


    def polygon_to_coords(self, label):
        poly = label['polygon']
        poly_coords = list(poly.exterior.coords)
        for coord in poly_coords:
            poly_coords[poly_coords.index(coord)] = list(coord)
        for num in range(len(poly_coords)):
            poly_coords[num] = list(map(int, poly_coords[num]))
        return poly_coords

    def to_histomicstk_heatmap_ann(self, wsi_dir, pred_ann, filename=""):
        """
        :param qupath_ann: json object for qupath annotaion
        :return: json file for histomicstk annotation
        https://github.com/girder/large_image/blob/master/girder_annotation/docs/annotations.md
        """

        elements = []
        # name = []

        for label in pred_ann:
            for l in pred_ann[label]:
                points = []
                for point in self.polygon_to_coords(l):
                    point.append(0)
                    points.append(point)
                # name.append(l['name'])

                elements_obj = {
                    "type": "polyline",
                    "label": {
                        "value": "%s" % l['name']
                    },
                    "lineColor": "rgb%s" % (ImageColor.getcolor(l['color'], "RGB"),),
                    "lineWidth": 1,
                    "closed": True,
                    "points": points,
                    "fillColor": "rgba%s" % (ImageColor.getcolor(l['color'], "RGB") + (l['conf'],),),

                }
                elements.append(elements_obj)

        # Init annotation document in DSA style
        histomicstk_ann = {'name': filename, 'description': '', 'elements': elements}

        with open(wsi_dir + '_{}_{}_htk.json'.format(self.cfg.TASK, filename), 'w') as f:
            json_dumps = json.dumps(histomicstk_ann, indent=2)
            f.write(json_dumps)

        return histomicstk_ann

    def to_histomicstk_ann(self, wsi_dir, pred_ann, filename=""):
        """
        :param qupath_ann: json object for qupath annotaion
        :return: json file for histomicstk annotation
        https://github.com/girder/large_image/blob/master/girder_annotation/docs/annotations.md
        """

        elements = []
        # name = []

        for label in pred_ann:
            for l in pred_ann[label]:
                points = []
                for point in self.polygon_to_coords(l):
                    point.append(0)
                    points.append(point)
                # name.append(l['name'])

                elements_obj = {
                    "type": "polyline",
                    "label": {
                        "value": "%s" % l['name']
                    },
                    "lineColor": "rgb%s" % (ImageColor.getcolor(l['color'], "RGB"),),
                    "lineWidth": 2,
                    "closed": True,
                    "points": points,
                    "fillColor": "rgb%s" % (ImageColor.getcolor(l['color'], "RGB"),),

                    # the official json file contains 'name' and 'label',
                    # check if those are necessary.
                }
                elements.append(elements_obj)

        # Init annotation document in DSA style
        histomicstk_ann = {'name': filename, 'description': '', 'elements': elements}

        with open(wsi_dir + '_{}_{}_htk.json'.format(self.cfg.TASK, filename), 'w') as f:
            json_dumps = json.dumps(histomicstk_ann, indent=2)
            f.write(json_dumps)

        return histomicstk_ann

    # Create function to apply a grey patch on an image
    def apply_grey_patch(self, img, top_left_x, top_left_y, patch_size):
        patched_image = np.array(img, copy=True)
        patched_image[top_left_y:top_left_y + patch_size, top_left_x:top_left_x + patch_size, :] = 127.5
        return patched_image
