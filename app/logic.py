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
import os
import shutil
import threading
import time

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np

jsonpickle_numpy.register_handlers()


class AppLogic:
    """ Implementing the workflow for FeatureCloud platform

    Attributes
    ----------
    status_available: bool
    status_finished: bool
    id:
    coordinator: bool
    clients:
    data_incoming: list
    data_outgoing: list
    iteration: int
    progress: str
    INPUT_DIR: str
    OUTPUT_DIR: str
    mode: str
    dir: str
    splits: dict
    test_splits: dict
    global_stats: dict
    iter_counter: int
    workflows_states: dict
    states: dict

    Methods
    -------
    handle_setup(client_id, coordinator, clients)
    handle_incoming(data)
    handle_outgoing()
    app_flow()
    send_to_server(data_to_send)
    wait_for_server()
    broadcast(data)
    read_config(config_file)
    read_input(path)
    local_preprocess(x_train, x_test, global_stats)
    global_aggregation(stats)
    write_results(train_set, test_set, output_path)
    lazy_initialization(mode, dir)
    finalize_config()

    """

    def __init__(self):

        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.preprocess = [False]

        # === States ===
        self.states = {"state_initializing": 1,
                       "state_load_images": 2,
                       "state_preprocess": 3,
                       "state_writing_results": 4,
                       "state_finishing": 5
                       }

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # Initial state
        state = self.states["state_initializing"]
        self.progress = 'initializing...'

        while True:
            print(f"{bcolors.STATE}{list(self.states.keys())[list(self.states.values()).index(state)]}{bcolors.ENDC}")
            if state == self.states["state_initializing"]:
                if self.id is not None:  # Test if setup has happened already
                    state = self.states["state_load_images"]

            if state == self.states["state_load_images"]:
                self.progress = "Config..."
                self.preprocess = self.read_config(self.INPUT_DIR + '/config.yml')
                print(self.preprocess)
                os.makedirs(self.INPUT_DIR.replace("/input", "/output"), exist_ok=True)
                shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')
                print(f'Read config file.', flush=True)
                self.load_images(self.INPUT_DIR)
                if any(self.preprocess):
                    state = self.states['state_preprocess']
                else:
                    state = self.states['state_writing_results']

            if state == self.states["state_preprocess"]:
                self.progress = 'Resizing images'
                self.image_preprocess(*self.preprocess)
                state = self.states['state_writing_results']

            if state == self.states["state_writing_results"]:
                self.progress = "write"
                self.write_results(self.OUTPUT_DIR)
                if self.coordinator:
                    self.data_incoming.append('DONE')
                    state = self.states["state_finishing"]
                else:
                    self.data_outgoing = 'DONE'
                    self.status_available = True

                    break

            if state == self.states["state_finishing"]:
                self.progress = 'finishing...'
                if len(self.data_incoming) == len(self.clients):
                    self.status_finished = True
                    break

            time.sleep(1)

    def send_to_server(self, data_to_send):
        """  Will be called only for clients
            to send their parameters or locally computed
             mean and standard deviation for the coordinator

        Parameters
        ----------
        data_to_send: list

        """
        data_to_send = jsonpickle.encode(data_to_send)
        if self.coordinator:
            self.data_incoming.append(data_to_send)
        else:
            self.data_outgoing = data_to_send
            self.status_available = True
            print(f'{bcolors.SEND_RECEIVE} [CLIENT] Sending data to coordinator. {bcolors.ENDC}', flush=True)

    def get_clients_data(self):
        """ Will be called only for the coordinator
            to get all the clients communicated data
            for each split, corresponding clients' data will be yield back.

        Returns
        -------
        clients_data: list
        split: str
        """
        print(f"{bcolors.SEND_RECEIVE} Received data of all clients. {bcolors.ENDC}")
        data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
        self.data_incoming = []
        for split in self.splits.keys():
            print(f'{bcolors.SPLIT} Get {split} {bcolors.ENDC}')
            clients_data = []
            for client in data:
                clients_data.append(client[split])
            yield clients_data, split

    def wait_for_server(self):
        """ Will be called only for clients
            to wait for server to get
            some globally shared data.

        Returns
        -------
        None or list
            in case no data received None will be returned
            to signal the state!
        """
        if len(self.data_incoming) > 0:
            data_decoded = jsonpickle.decode(self.data_incoming[0])
            self.data_incoming = []
            return data_decoded
        return None

    def broadcast(self, data):
        """ will be called only for the coordinator after
            providing data that should be broadcast to clients

        Parameters
        ----------
        data: list

        """
        data_to_broadcast = jsonpickle.encode(data)
        self.data_outgoing = data_to_broadcast
        self.status_available = True
        print(f'{bcolors.SEND_RECEIVE} [COORDINATOR] Broadcasting data to clients. {bcolors.ENDC}', flush=True)

    def read_config(self, config_file):
        """ should be overridden!
            reads the config file.

        Parameters
        ----------
        config_file: string
            path to the config.yaml file!

        Returns
        -------
        list_of_booleans: list
            : including boolean values for different
            optional preprocesses.

        Raises
        ------
        NotImplementedError
        """
        NotImplementedError("read_config method in Applogic class is not implemented!")

    def load_images(self, path):
        """ should be overridden
        load images and read their labels from the labels file
         or label images based on their folder name.

        Parameters
        ----------
        path: str
            for one application it would be "/mnt/input"


        Returns
        -------

        Raises
        ------
        NotImplementedError
        """
        NotImplementedError("preprocess method in Applogic class is not implemented!")

    def image_preprocess(self, resize, crop):
        """ should be overridden!
            called for clients to normalized their data based on global statistics.

        Parameters
        ----------
        resize: bool
        crop: bool

        Returns
        -------

        Raises
        ------
        NotImplementedError
        """

        NotImplementedError("preprocess method in Applogic class is not implemented!")

    def write_results(self, output_path):
        """ should be overridden!
            writing loaded and preprocessed images as .npy file.

        Parameters
        ----------
        output_path: str

        Raises
        ------
        NotImplementedError
        """
        NotImplementedError("write_results method in Applogic class is not implemented!")


class TextColor:
    def __init__(self, color):
        if color:
            self.SEND_RECEIVE = '\033[95m'
            self.STATE = '\033[94m'
            self.SPLIT = '\033[96m'
            self.VALUE = '\033[92m'
            self.WARNING = '\033[93m'
            self.FAIL = '\033[91m'
            self.ENDC = '\033[0m'
            self.BOLD = '\033[1m'
            self.UNDERLINE = '\033[4m'
        else:
            self.SEND_RECEIVE = ''
            self.STATE = ''
            self.SPLIT = ''
            self.VALUE = ''
            self.WARNING = ''
            self.FAIL = ''
            self.ENDC = ''
            self.BOLD = ''
            self.UNDERLINE = ''


bcolors = TextColor(color=False)
