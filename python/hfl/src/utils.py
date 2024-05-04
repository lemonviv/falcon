
import argparse
import pickle
import socket
import struct

import torch
from google.protobuf.message import Message
from torch import tensor
from .sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from .sampling import cifar_iid, cifar_noniid
from torchvision import datasets, transforms

def receive_all(conn: socket.socket, size: int) -> bytes:
    """Receive a given length of bytes from socket.

    Args:
        conn (socket.socket): Socket connection.
        size (int): Length of bytes to receive.

    Raises:
        RuntimeError: If connection closed before chunk was read, it will raise an error.

    Returns:
        bytes: Received bytes.
    """
    buffer = b""
    while size > 0:
        chunk = conn.recv(size)
        if not chunk:
            raise RuntimeError("connection closed before chunk was read")
        buffer += chunk
        size -= len(chunk)
    return buffer


def send_int(conn: socket.socket, i: int, pack_format: str = "Q") -> None:
    """Send an integer from socket.

    Args:
        conn (socket.socket): Socket connection.
        i (int): Integer to send.
        pack_format (str, optional): Pack format. Defaults to "Q", which means unsigned long long.
    """
    data = struct.pack(f"!{pack_format}", i)
    conn.sendall(data)


def receive_int(conn: socket.socket, pack_format: str = "Q") -> int:
    """Receive an integer from socket.

    Args:
        conn (socket.socket): Socket connection.
        pack_format (str, optional): Pack format. Defaults to "Q", which means unsigned long long.

    Returns:
        int: Received integer.
    """
    buffer_size = struct.Struct(pack_format).size
    data = receive_all(conn, buffer_size)
    (data,) = struct.unpack(f"!{pack_format}", data)
    return data


def send_message(conn: socket.socket, data: Message, pack_format: str = "Q") -> None:
    """Send protobuf message from socket. First the length of protobuf message will be sent. Then the message is sent.

    Args:
        conn (socket.socket): Socket connection.
        data (Message): Protobuf message to send.
        pack_format (str, optional): Length of protobuf message pack format. Defaults to "Q", which means unsigned long long.
    """
    send_int(conn, data.ByteSize(), pack_format)
    conn.sendall(data.SerializePartialToString())


def receive_message(conn: socket.socket, data: Message, pack_format: str = "Q") -> Message:
    """Receive protobuf message from socket

    Args:
        conn (socket.socket): Socket connection.
        data (Message): Placehold for protobuf message.
        pack_format (str, optional): Length of protobuf message pack format. Defaults to "Q", which means unsigned long long.

    Returns:
        Message: The protobuf message.
    """
    data_len = receive_int(conn, pack_format)
    data.ParseFromString(receive_all(conn, data_len))
    return data


def serialize_tensor(t: torch.tensor) -> bytes:
    """Serialize a torch tensor to bytes.

    Args:
        t (torch.tensor): The torch tensor.

    Returns:
        bytes: The serialized tensor.
    """
    numpy_t = t.numpy()
    return pickle.dumps(numpy_t,)


def deserialize_tensor(t: bytes) -> torch.tensor:
    """Recover torch tensor from bytes.

    Args:
        t (bytes): The serialized tensor.

    Returns:
        torch.tensor: The torch tensor.
    """
    return torch.from_numpy(pickle.loads(t))


def get_dataset(args):
    """ Returns train and test datasets.
    """

    if args.data == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.data_dist == 'iid':
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_clients)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_clients)

    elif args.data == 'mnist' or 'fmnist':
        if args.data == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.data_dist == 'iid':
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_clients)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_clients)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_clients)

    return train_dataset, test_dataset, user_groups

def parseargs(arg=None) -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        argparse.Namespace: parsed arguments
    """

    parser = argparse.ArgumentParser(description="Training using the autograd and graph.")
    parser.add_argument(
        "--model", choices=["cnn", "resnet", "xceptionnet", "mlp", "alexnet"], default="mlp"
    )
    parser.add_argument("--data", choices=["mnist", "cifar10", "cifar100", "bank"], default="mnist")
    parser.add_argument(
        "-m", "--max-epoch", default=10, type=int, help="maximum epochs", dest="max_epoch"
    )
    parser.add_argument(
        "-b", "--batch-size", default=64, type=int, help="batch size", dest="batch_size"
    )
    parser.add_argument(
        "-l", "--learning-rate", default=0.005, type=float, help="initial learning rate", dest="lr"
    )
    # Determine which gpu to use
    parser.add_argument(
        "-i", "--device-id", default=0, type=int, help="which GPU to use", dest="device_id"
    )
    parser.add_argument(
        "-g",
        "--disable-graph",
        default="True",
        action="store_false",
        help="disable graph",
        dest="graph",
    )
    parser.add_argument(
        "-v", "--log-verbosity", default=0, type=int, help="logging verbosity", dest="verbosity"
    )
    parser.add_argument(
        "-d",
        "--data-distribution",
        choices=["iid", "non-iid"],
        default="iid",
        help="data distribution",
        dest="data_dist",
    )
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                                of channels of imgs")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                            of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                            to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                            of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument("--num_clients", default=10, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=1234)
    
    args = parser.parse_args(arg)
    return args
