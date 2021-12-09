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
from FeatureCloud.engine.app import app_state, AppState, Role, LogLevel
from FeatureCloud.engine.app import State as op_state
import pandas as pd
import numpy as np
import os
from PIL import Image
import glob
import ConfigState

name = 'image_loader'


@app_state(name='initial', role=Role.BOTH, app_name=name)
class ImgLoader(ConfigState.State):

    def register(self):
        self.register_transition('WriteResults', Role.BOTH)

    def run(self):
        self.lazy_init()
        self.read_config()
        self.finalize_config()
        samples, labels = self.load_images(ds_dir=f"{self.input_dir}/{self.config['local_dataset']['ds_dir']}")
        self.store('labels', labels)
        self.update(progress=0.3)
        self.store('samples', self.image_preprocess(samples))
        self.update(progress=0.8)
        return 'WriteResults'

    def load_images(self, ds_dir):
        image_formats = self.config['local_dataset']['image_format']
        if not isinstance(image_formats, list):
            self.log(f"image formats should listed; even there is only one item!\n"
                     f"e.g., [jpeg]", LogLevel.ERROR)
            self.update(state=op_state.ERROR)
        self.log(f"Image formats: {image_formats}")
        self.log(f"Reading {ds_dir}", LogLevel.DEBUG)
        self.update(progress=0.1)
        samples = []
        labels = []
        target = self.config['local_dataset']['target_value']
        if '.txt' in target or '.csv' in target:
            folders = []
            for folder in glob.glob(f'{ds_dir}/*/'):
                folders.append(folder.strip().split('/')[-2])
            for folder in folders:
                self.log(f"Loading {folder}...")
                labels_file = f"{ds_dir}/folder/{target}"
                if os.path.exists(labels_file):
                    df = pd.read_csv(labels_file, sep=self.config['local_dataset']['sep'])
                else:
                    self.update(state=op_state.ERROR)
                    self.log(f"No {target} file found in {labels_file}!",
                                 LogLevel.FATAL)
                for format in image_formats:
                    for filename in glob.glob(f'{ds_dir}/{folder}/*.{format}'):  # assuming gif
                        samples.append(Image.open(filename))
                        labels.append(df[df.name == filename.strip().split('/')[-1]].label.values.item())
            return samples, labels

        labels_folders = []
        for folder in glob.glob(f'{ds_dir}/*/'):
            labels_folders.append(folder.strip().split('/')[-2])
        for folder in labels_folders:
            self.log(f"Loading {folder}...")
            for format in image_formats:
                for filename in glob.glob(f'{ds_dir}/{folder}/*.{format}'):  # assuming gif
                    samples.append(Image.open(filename))
                    labels.append(folder)
        return samples, labels

    def image_preprocess(self, samples):
        resize, crop = self.config.get('image_resize', False), self.config.get('image_crop', False)
        if resize:
            resize_dim = (resize['width'], resize['height'])
            resized_samples = []
            for sample in samples:
                resized_samples.append(sample.resize(resize_dim))
            samples = resized_samples
        if crop:
            crop_sizes = (crop['x_coordinate'],
                          crop['y_coordinate'],
                          crop['width'],
                          crop['height'])
            cropped_samples = []
            for sample in samples:
                cropped_samples.append(sample.crop(crop_sizes))
            samples = cropped_samples
        return samples


@app_state(name='WriteResults', role=Role.BOTH)
class WriteResults(AppState):
    def register(self):
        self.register_transition('terminal', Role.BOTH)

    def run(self) -> str or None:
        samples = []
        for sample in self.load('samples'):
            samples.append(np.asarray(sample))
        np.save(self.load('output_files')['data'][0], [samples, self.load('labels')])
        self.update(progress=0.99)
        return 'terminal'
