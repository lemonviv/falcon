
# modified from https://github.com/zhengzangw/Fed-SINGA/blob/main/src/server/app.py

import socket
from collections import defaultdict
from typing import Dict, List

import torch
import risefl_interface

from .proto import interface_pb2 as proto
from .utils import (parseargs, receive_int, receive_all, receive_message, send_message, send_int,
                    serialize_tensor, deserialize_tensor, check_defense_type, send_string, receive_string, unpack_flatten_model_weights)
from .models import MLP_Bank, CNNMnist
import os
import base64
import numpy as np


class Server:
    """Server sends and receives protobuf messages.

    Create and start the server, then use pull and push to communicate with clients.

    Attributes:
        num_clients (int): Number of clients.
        host (str): Host address of the server.
        port (str): Port of the server.
        sock (socket.socket): Socket of the server.
        conns (List[socket.socket]): List of num_clients sockets.
        addrs (List[str]): List of socket address.
        weights (Dict[Any]): Weights stored on server.
    """

    def __init__(
        self,
        num_clients=1,
        host: str = "127.0.0.1",
        port: str = 1234,
    ) -> None:
        """Class init method

        Args:
            num_clients (int, optional): Number of clients in training.
            host (str, optional): Host ip address. Defaults to '127.0.0.1'.
            port (str, optional): Port. Defaults to 1234.
        """
        self.num_clients = num_clients
        self.host = host
        self.port = port

        self.sock = socket.socket()
        self.conns = [None] * num_clients
        self.addrs = [None] * num_clients

        self.weights = {}

        self.zkp_server = None
        self.check_param = None

    def __start_connection(self) -> None:
        """Start the network connection of server."""
        self.sock.bind((self.host, self.port))
        self.sock.listen()
        print("Server started.")

    def __start_rank_pairing(self) -> None:
        """Start pair each client to a global rank"""
        for _ in range(self.num_clients):
            conn, addr = self.sock.accept()
            rank = receive_int(conn)
            self.conns[rank] = conn
            self.addrs[rank] = addr
            print(f"[Server] Connected by {addr} [global_rank {rank}]")

        assert None not in self.conns

    def init_server_zkp(self, args) -> None:
        """Initialize server zkp"""
        defense_type = check_defense_type(args.check_type)

        # initialize the check parameter
        self.check_param = risefl_interface.CheckParamFloat(defense_type)
        self.check_param.l2_param.bound = args.norm_bound

        # initialize server
        self.zkp_server = risefl_interface.ServerInterface(
            args.num_clients, args.max_malicious_clients, args.dim,
            args.num_blinds_per_group_element, args.weight_bits, args.random_normal_bit_shifter,
            args.num_norm_bound_samples, args.inner_prod_bound_bits, args.max_bound_sq_bits,
            defense_type, False)

        # TODO: need to broadcast random_bytes_str to all clients
        # a random string used to generate independent group elements, to be used by both the server and clients
        # random_bytes = os.urandom(64)
        # random_bytes_str = base64.b64encode(random_bytes).decode('ascii')
        random_bytes_str = "r0sdTz/eXbBDsPpB9QiB4P+ejll9juZdbYa4Xt+OZbFlV/n7FUcTMas64getSoWMoV5hE+UmiR6W554xa4SPnQ=="
        print("random_bytes_str = " + random_bytes_str)

        self.zkp_server.initialize_from_seed(random_bytes_str)

    def start(self) -> None:
        """Start the server.

        This method will first bind and listen on the designated host and port.
        Then it will connect to num_clients clients and maintain the socket.
        In this process, each client shall provide their rank number.
        """
        self.__start_connection()
        self.__start_rank_pairing()

    def close(self) -> None:
        """Close the server."""
        self.sock.close()

    def aggregate(self, weights: Dict[str, List[torch.tensor]]) -> Dict[str, torch.tensor]:
        """Aggregate collected weights to update server weight.

        Args:
            weights (Dict[str, List[torch.tensor]]): The collected weights.

        Returns:
            Dict[str, torch.tensor]: Updated weight stored in server.
        """
        for k, v in weights.items():
            self.weights[k] = sum(v) / self.num_clients
        return self.weights

    # def average_weights(w):
    #     """
    #     Returns the average of the weights.
    #     """
    #     w_avg = copy.deepcopy(w[0])
    #     for key in w_avg.keys():
    #         for i in range(1, len(w)):
    #             w_avg[key] += w[i][key]
    #         w_avg[key] = torch.div(w_avg[key], len(w))
    #     return w_avg

    def pull(self) -> None:
        """Server pull weights from clients.

        Namely clients push weights to the server. It is the gather process.
        """
        # open space to collect weights from clients
        datas = [proto.WeightsExchange() for _ in range(self.num_clients)]
        weights = defaultdict(list)
        # receive weights sequentially
        for i in range(self.num_clients):
            datas[i] = receive_message(self.conns[i], datas[i])
            for k, v in datas[i].weights.items():
                weights[k].append(deserialize_tensor(v))
        # aggregation
        self.aggregate(weights)

    def push(self) -> None:
        """Server push weights to clients.

        Namely clients pull weights from server. It is the scatter process.
        """
        message = proto.WeightsExchange()
        message.op_type = proto.SCATTER
        for k, v in self.weights.items():
            message.weights[k] = serialize_tensor(v)

        for conn in self.conns:
            send_message(conn, message)


if __name__ == "__main__":
    args = parseargs()

    server = Server(num_clients=args.num_clients, host=args.host, port=args.port)
    server.start()

    # TODO: need to start a bulletin to generate these keys, or let each client generate and register
    # before the training start, the server generates the pub key and priv key and
    # send them to the corresponding clients
    sign_pub_keys_vec = risefl_interface.VecSignPubKeys(args.num_clients + 1)
    sign_prv_keys_vec = risefl_interface.VecSignPrvKeys(args.num_clients + 1)
    # init the pub_keys_vec and prv_keys_vec
    for i in range(args.num_clients + 1):
        sign_key_pair = risefl_interface.gen_sign_key_pair()
        sign_pub_keys_vec[i] = sign_key_pair.first
        sign_prv_keys_vec[i] = sign_key_pair.second

    # need to update to key exchange (the current implementation is not efficient)
    # send the pub_keys_vec and prv_keys_vec[i] to client i, for i \in [1, num_clients + 1]
    for i in range(args.num_clients):
        # send the sign_pub_keys_vec and the corresponding sign_prv_keys[i] to each client
        print("i = ", i)
        for j in range(args.num_clients + 1):
            print("j = ", j)
            sign_pub_keys_vec_j_str = risefl_interface.convert_sign_pub_key_to_string(sign_pub_keys_vec[j])
            print(f"sign_pub_keys_vec_j_str: {sign_pub_keys_vec_j_str}")
            print(f"size = {len(sign_pub_keys_vec_j_str)}")
            send_string(server.conns[i], sign_pub_keys_vec_j_str)
            print("send")
        # serialize the private key and send it the client i
        # as the client index of the sign_prv_keys_vec start from 1
        sign_prv_keys_vec_i_str = risefl_interface.convert_sign_prv_key_to_string(sign_prv_keys_vec[i+1])
        print(f"sign_prv_keys_vec_i_str: {sign_prv_keys_vec_i_str}")
        print(f"size = {len(sign_prv_keys_vec_i_str)}")
        send_string(server.conns[i], sign_prv_keys_vec_i_str)
        print("send")

    server.init_server_zkp(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.data == 'mnist':
            global_model = CNNMnist(num_channels=args.num_channels, num_classes=args.num_classes)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        # TODO: need to get from argument or client
        img_size = torch.Size([1, 63])
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP_Bank(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    for i in range(args.max_epoch):
        print(f"On epoch {i}:")
        if i > 0:
            # Push to Clients
            print(f"Server push weights to clients start")
            server.push()
            print(f"Server push weights to clients done")

        # Collects from Clients
        # print(f"Server pull weights from clients start")
        # server.pull()
        # print(f"Server pull weights from clients done")

        # add the following to the iterations
        server.zkp_server.initialize_new_iteration(server.check_param)

        print("server initialize new iteration finished")

        # step 1 receive messages from all clients
        for i in range(args.num_clients):
            # receive messages from each client
            client_send_str1_i = receive_string(server.conns[i])
            server.zkp_server.receive_1(client_send_str1_i, i + 1)

        print("server step 1 finished")

        # step 2 send messages to all clients and receive messages from all clients
        bytes_sent_2 = server.zkp_server.send_2()
        for i in range(args.num_clients):
            # broadcast the message
            send_string(server.conns[i], bytes_sent_2)

        print("server step 2 -- send string finished")

        # what is the difference with two loops and one loop
        for i in range(args.num_clients):
            # receive the message from the client
            client_send_str2_i = receive_string(server.conns[i])
            server.zkp_server.receive_2(client_send_str2_i, i + 1)

        print("server step 2 -- receive string finished")

        # step 3 send messages to all clients and receive messages
        server.zkp_server.concurrent_process_before_send_3()
        for i in range(args.num_clients):
            server_send_3_str = server.zkp_server.send_3(i)
            # send the string to client i
            send_string(server.conns[i], server_send_3_str)

        print("server step 3 -- send string finished")

        for i in range(args.num_clients):
            # receive str from client i
            client_send_str3_i = receive_string(server.conns[i])
            server.zkp_server.receive_3(client_send_str3_i, i + 1)

        print("server step 3 -- receive string finished")

        # step 4 send messages to all clients and receive
        server.zkp_server.process_before_send_4()
        for i in range(args.num_clients):
            server_send_4_str = server.zkp_server.send_4(i)
            # send string to client i
            send_string(server.conns[i], server_send_4_str)

        print("server step 4 -- send string finished")

        for i in range(args.num_clients):
            # receive string from client i
            client_send_str4 = receive_string(server.conns[i])
            server.zkp_server.receive_4(client_send_str4, i + 1)

        print("server step 4 -- receive string finished")

        # step 5 send messages to all clients and receive
        server.zkp_server.process_before_send_5()
        for i in range(args.num_clients):
            server_send_5_str = server.zkp_server.send_5(i)
            # send string to client i
            send_string(server.conns[i], server_send_5_str)

        print("server step 5 -- send string finished")

        for i in range(args.num_clients):
            # receive string from client i
            client_send_str5 = receive_string(server.conns[i])
            server.zkp_server.receive_5(client_send_str5, i + 1)

        print("server step 5 -- receive string finished")

        # finish one iteration
        server.zkp_server.finish_iteration()

        print("server finish iteration")

        # reconstruct the flattened weights to tensors and assign to server.weights
        print(f"final_update.type: {type(server.zkp_server.final_update_float)}")
        final_update_float_ndarray = np.array(server.zkp_server.final_update_float)
        weights = unpack_flatten_model_weights(global_model, final_update_float_ndarray)
        server.weights = weights

    server.close()
