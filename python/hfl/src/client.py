#!/usr/bin/env python3

import socket

from .proto import interface_pb2 as proto
from .utils import parseargs, get_dataset

import numpy as np

import copy
import torch
from tqdm import tqdm

from . import bank, utils
from .models import MLP_Bank, CNNMnist

from .update import LocalUpdate, test_inference


class Client:
    """Client sends and receives protobuf messages.

    Create and start the server, then use pull and push to communicate with the server.

    Attributes:
        global_rank (int): The rank in training process.
        host (str): Host address of the server.
        port (str): Port of the server.
        sock (socket.socket): Socket of the client.
        weights (Dict[Any]): Weights stored locally.
    """

    def __init__(
        self,
        global_rank: int = 0,
        host: str = "127.0.0.1",
        port: str = 1234,
    ) -> None:
        """Class init method

        Args:
            global_rank (int, optional): The rank in training process. Defaults to 0.
            host (str, optional): Host ip address. Defaults to '127.0.0.1'.
            port (str, optional): Port. Defaults to 1234.
        """
        self.host = host
        self.port = port
        self.global_rank = global_rank

        self.sock = socket.socket()

        self.weights = {}

    def __start_connection(self) -> None:
        """Start the network connection to server."""
        self.sock.connect((self.host, self.port))

    def __start_rank_pairing(self) -> None:
        """Sending global rank to server"""
        utils.send_int(self.sock, self.global_rank)

    def start(self) -> None:
        """Start the client.

        This method will first connect to the server. Then global rank is sent to the server.
        """
        self.__start_connection()
        self.__start_rank_pairing()

        print(f"[Client {self.global_rank}] Connect to {self.host}:{self.port}")

    def close(self) -> None:
        """Close the server."""
        self.sock.close()

    def pull(self) -> None:
        """Client pull weights from server.

        Namely server push weights from clients.
        """
        message = proto.WeightsExchange()
        message = utils.receive_message(self.sock, message)
        for k, v in message.weights.items():
            self.weights[k] = utils.deserialize_tensor(v)

    def push(self) -> None:
        """Client push weights to server.

        Namely server pull weights from clients.
        """
        message = proto.WeightsExchange()
        message.op_type = proto.GATHER
        for k, v in self.weights.items():
            message.weights[k] = utils.serialize_tensor(v)
        utils.send_message(self.sock, message)


# Calculate accuracy
def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


# Data partition according to the rank
def partition(global_rank, world_size, train_x, train_y, val_x, val_y):
    # Partition training data
    data_per_rank = train_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    train_x = train_x[idx_start:idx_end]
    train_y = train_y[idx_start:idx_end]

    # Partition evaluation data
    data_per_rank = val_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    val_x = val_x[idx_start:idx_end]
    val_y = val_y[idx_start:idx_end]
    return train_x, train_y, val_x, val_y



def get_data(data, data_dist="iid", device_id=None):
    if data == "bank":
        train_x, train_y, val_x, val_y, num_classes = bank.load(device_id)
    else:
        raise NotImplementedError
    return train_x, train_y, val_x, val_y, num_classes


def get_model(model, num_channels=None, num_classes=None, data_size=None):
    if model == "mlp":
        model = MLP_Bank(dim_in=data_size, dim_hidden=64, dim_out=num_classes)
    else:
        raise NotImplementedError
    return model


def run(
        args,
        global_rank,
        world_size,
        device_id,
        max_epoch,
        batch_size,
        model,
        data,
        data_dist,
        graph,
        verbosity,
        spars=None
):
    # Connect to server
    client = Client(global_rank=device_id)
    client.start()

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.data == 'mnist':
            global_model = CNNMnist(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP_Bank(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()
    client.weights = global_weights

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.max_epoch)):
        # local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        if epoch > 0:
            client.pull()

        m = args.num_clients
        idxs_users = np.random.choice(range(args.num_clients), m, replace=False)


        for idx in idxs_users:
            if idx == client.global_rank:
                # update global weights
                global_model.load_state_dict(client.weights)

                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx])
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
                # local_weights.append(copy.deepcopy(w))
                # local_losses.append(copy.deepcopy(loss))

        client.weights = copy.deepcopy(w)
        print(f'Local training loss : {loss}')
        client.push()

        # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.max_epoch} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))


    client.close()


if __name__ == "__main__":
    args = parseargs()
    # sgd = opt.SGD(lr=args.lr, momentum=0.9, weight_decay=1e-5, dtype=singa_dtype[args.precision])

    run(
        args,
        0,
        1,
        args.device_id,
        args.max_epoch,
        args.batch_size,
        args.model,
        args.data,
        args.data_dist,
        args.graph,
        args.verbosity
    )
