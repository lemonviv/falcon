#!/usr/bin/env python3

import socket

from .proto import interface_pb2 as proto
from .utils import parseargs, check_defense_type, receive_message, send_message, send_string, receive_string
from .dataset.data_loader import get_dataset

import numpy as np
import copy
import os
import base64
from tqdm import tqdm

from . import utils
from .dataset import bank
from .models import MLP_Bank, CNNMnist

from .update import LocalUpdate, test_inference

import risefl_interface

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

        self.zkp_client = None
        self.check_param = None

    def __start_connection(self) -> None:
        """Start the network connection to server."""
        self.sock.connect((self.host, self.port))

    def __start_rank_pairing(self) -> None:
        """Sending global rank to server"""
        utils.send_int(self.sock, self.global_rank)

    def init_zkp_client(self, args) -> None:
        """Initialize zkp params etc."""
        defense_type = check_defense_type(args.defense_desc)

        sign_pub_keys_vec = risefl_interface.VecSignPubKeys(args.num_clients + 1)
        for j in range(args.num_clients + 1):
            recv_pub_key_j_str = receive_string(self.sock)
            sign_pub_keys_vec[j] = risefl_interface.convert_string_to_sign_pub_key(recv_pub_key_j_str)
        recv_prv_key_str = receive_string(self.sock)
        sign_prv_keys_vec_i = risefl_interface.convert_string_to_sign_prv_key(recv_prv_key_str)

        # the client id in the zkp library starts from 1, so the index needs to be increased by 1
        self.zkp_client = risefl_interface.ClientInterface(
            args.num_clients, args.max_malicious_clients, args.dim,
            args.num_blinds_per_group_element, args.weight_bits, args.random_normal_bit_shifter,
            args.num_norm_bound_samples, args.inner_prod_bound_bits, args.max_bound_sq_bits,
            defense_type, self.global_rank + 1,
            sign_pub_keys_vec, sign_prv_keys_vec_i)

        # initialize the check parameter
        self.check_param = risefl_interface.CheckParamFloat(defense_type)
        self.check_param.l2_param.bound = args.norm_bound

        # TODO: need to receive random_bytes_str from server, currently hard-coded for both server and clients
        # a random string used to generate independent group elements, to be used by both the server and clients
        # random_bytes = os.urandom(64)
        # random_bytes_str = base64.b64encode(random_bytes).decode('ascii')
        random_bytes_str = "r0sdTz/eXbBDsPpB9QiB4P+ejll9juZdbYa4Xt+OZbFlV/n7FUcTMas64getSoWMoV5hE+UmiR6W554xa4SPnQ=="
        print("random_bytes_str = " + random_bytes_str)

        self.zkp_client.initialize_from_seed(random_bytes_str)

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
        verbosity,
        spars=None
):
    # Connect to server
    client = Client(global_rank=device_id)
    client.start()

    client.init_zkp_client(args)

    # if args.gpu_id:
    #     torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset = get_dataset(args.data, args.data_dir, client_id=client.global_rank)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.data == 'mnist':
            global_model = CNNMnist(num_channels=args.num_channels, num_classes=args.num_classes)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP_Bank(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
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

    for epoch in tqdm(range(args.max_epoch)):
        # local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        if epoch > 0:
            print(f"client begins to pull weights from server")
            client.pull()
            print(f"client finishes pulling weights from server")

        global_model.load_state_dict(client.weights)

        print(f"client start local update...")
        local_model = LocalUpdate(args=args, dataset=train_dataset)
        w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
        print(f"client finish local update...")

        client.weights = copy.deepcopy(w)
        print(f'Local training loss : {loss}')

        # step 1 client sends message to the server
        # flatten weights to 1D array
        flatten_weights = client.weights
        converted_weights = risefl_interface.VecFloat(flatten_weights)
        client_send_str1 = client.zkp_client.send_1(client.check_param, converted_weights)
        # send this string to the server
        send_string(client.sock, client_send_str1)

        # step 2 receive message from the server and sends message back to the server
        sent_2 = receive_string(client.sock)
        bytes_sent_2 = sent_2.encode()
        client_send_str2 = client.zkp_client.receive_and_send_2(bytes_sent_2)
        send_string(client.sock, client_send_str2)

        # step 3 receive message from the server and sends message back to the server
        # receive message from server
        server_sent_3_str = receive_string(client.sock)
        client_send_str3 = client.zkp_client.receive_and_send_3(server_sent_3_str)
        send_string(client.sock, client_send_str3)

        # step 4 receive message from the server and sends message back to the server
        # receive message from server
        server_sent_4_str = receive_string(client.sock)
        client_send_str4 = client.zkp_client.receive_and_send_4(server_sent_4_str)
        send_string(client.sock, client_send_str4)

        # step 5 receive message from the server and sends message back to the server
        # receive message from server
        server_sent_5_str = receive_string(client.sock)
        client_send_str5 = client.zkp_client.receive_and_send_5(server_sent_5_str)
        send_string(client.sock, client_send_str5)

        # make changes in local_model

        # client.push()
        # print(f"client pushed weights to server")

        local_model = LocalUpdate(args=args, dataset=train_dataset)
        acc, loss = local_model.inference(model=global_model)
        train_accuracy.append(acc)
        train_loss.append(loss)
        print("|---- Train Accuracy: {:.2f}%".format(100*acc))
        print("|---- Train Loss: {:.2f}".format(loss))

    print(f"Train Accuracy: {train_accuracy}")
    print(f"Train Loss: {train_loss}")

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.max_epoch} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
    print("|---- Test Loss: {:.2f}".format(test_loss))

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
        args.verbosity
    )
