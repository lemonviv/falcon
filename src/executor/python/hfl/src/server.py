
# modified from https://github.com/zhengzangw/Fed-SINGA/blob/main/src/server/app.py

import socket
from collections import defaultdict
from typing import Dict, List

import torch

from .proto import interface_pb2 as proto
from .utils import parseargs, receive_int, receive_all, receive_message, send_message, send_int, serialize_tensor, deserialize_tensor


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

    for i in range(args.max_epoch):
        print(f"On epoch {i}:")
        if i > 0:
            # Push to Clients
            print(f"Server push weights to clients start")
            server.push()
            print(f"Server push weights to clients done")

        # Collects from Clients
        print(f"Server pull weights from clients start")
        server.pull()
        print(f"Server pull weights from clients done")

    server.close()
