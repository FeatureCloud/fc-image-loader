"""
    FeatureCloud Image Loader Application

    Copyright 2021 Mohammad Bakhtiari. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
from .logic import AppLogic, bcolors
import numpy as np
import bios
import os
from PIL import Image
import pandas as pd
import glob


class CustomLogic(AppLogic):
    """ Subclassing AppLogic for overriding specific methods
        to implement the Image Normalization application.

    Attributes
    ----------
    train_filename: str
    test_filename: str
    stats: dict
        statistics of the dataset.
    method: str

    Methods
    -------
    read_config(config_file)
    read_input(path)
    local_preprocess(x_train, x_test, global_stats)
    global_aggregation(stats)
    write_results(train_set, test_set, output_path)
    """

    def __init__(self):
        super().__init__()
        self.ds_dir = None
        self.img_format = None
        self.labels_filename = None
        self.samples, self.labels = [], []
        self.resize_dim = None
        self.crop_sizes = None

    def read_config(self, config_file):
        config = bios.read(config_file)['fc_image_loader']
        self.ds_dir = config['ds_dir']
        self.img_format = config['image_format']
        self.labels_filename = config['labels']
        resize, crop = False, False
        if config['image_resize'] is not None:
            self.resize_dim = (config['image_resize']['width'], config['image_resize']['height'])
            resize = True
        if config['image_crop']:
            self.crop_sizes = (config['image_crop']['x_coordinate'],
                               config['image_crop']['y_coordinate'],
                               config['image_crop']['width'],
                               config['image_crop']['height'])
            crop = True
        return [resize, crop]

    def load_images(self, path):
        self.ds_dir = path + "/" + self.ds_dir
        print(f"{bcolors.VALUE}Reading {self.ds_dir} ...{bcolors.ENDC}")
        if '.txt' in self.labels_filename or '.csv' in self.labels_filename:
            folders = []
            for folder in glob.glob(f'{self.ds_dir}/*/'):
                folders.append(folder.strip().split('/')[-2])
            for folder in folders:
                labels_file = f"{self.ds_dir}/folder/{self.labels_filename}"
                if os.path.exists(labels_file):
                    df = pd.read_csv(labels_file, sep=self.sep)
                else:
                    raise FileNotFoundError(f"No {self.labels_filename} file found in {labels_file}!")
                for format in self.img_format:
                    for filename in glob.glob(f'{self.ds_dir}/{folder}/*.{format}'):  # assuming gif
                        self.samples.append(Image.open(filename))
                        self.labels.append(df[df.name == filename.strip().split('/')[-1]].label.values.item())
        else:  # labels on folder name
            labels_folders = []
            for folder in glob.glob(f'{self.ds_dir}/*/'):
                labels_folders.append(folder.strip().split('/')[-2])
            for folder in labels_folders:
                for format in self.img_format:
                    for filename in glob.glob(f'{self.ds_dir}/{folder}/*.{format}'):  # assuming gif
                        self.samples.append(Image.open(filename))
                        self.labels.append(folder)

    def image_preprocess(self, resize, crop):
        if resize:
            samples = []
            for sample in self.samples:
                samples.append(sample.resize(self.resize_dim))
            self.samples = samples
        if resize:
            samples = []
            for sample in self.samples:
                samples.append(sample.crop(self.crop_sizes))
            self.samples = samples

    def write_results(self, output_path):
        samples = []
        for sample in self.samples:
            samples.append(np.asarray(sample))
        np.save(f'{output_path}/dataset.npy', [samples, self.labels])


logic = CustomLogic()
