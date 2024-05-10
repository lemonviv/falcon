
import argparse
import pickle
import socket
import struct

import torch
import numpy as np
from google.protobuf.message import Message

import risefl_interface


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


def send_string(conn: socket.socket, data: str) -> None:
    data_bytes = data.encode()
    data_bytes_len = len(data_bytes)
    send_int(conn, data_bytes_len)
    conn.sendall(data_bytes)


def receive_string(conn: socket.socket) -> str:
    data_len = receive_int(conn)
    print("****** [utils.receive_string] data_len = ", data_len)
    data = receive_all(conn, data_len)
    # print("received_string_bytes = ", data)
    data_str = data.decode()
    # print("data_str = ", data_str)
    return data_str


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


def check_defense_type(check_type) -> int:
    defense_type = 0
    if check_type == 0:
        defense_type = risefl_interface.CHECK_TYPE_L2NORM
    elif check_type == 1:
        defense_type = risefl_interface.CHECK_TYPE_SPHERE
    elif check_type == 2:
        defense_type = risefl_interface.CHECK_TYPE_COSINE_SIM
    else:
        raise ValueError("Unsupported defense type!")
    return defense_type


def flatten_model_weights(model_weights):
    # Get the model parameters as a dictionary
    # state_dict = model.state_dict()

    # Flatten all parameters into a 1D array
    flattened_weights = np.concatenate([p.flatten().cpu().numpy() for p in model_weights.values()])
    return flattened_weights


def unpack_flatten_model_weights(model, flattened_weights):
    # Get the model parameters as a dictionary
    state_dict = model.state_dict()

    # Start index to keep track of the current position in the flattened array
    start_index = 0

    # Loop through the parameters in the state dict
    for key, value in state_dict.items():
        # Calculate the size of the parameter tensor
        size = np.prod(value.size())
        # Extract the corresponding segment from the flattened array
        param_data = flattened_weights[start_index:start_index+size]
        # Reshape the data and convert it to a tensor
        param_tensor = torch.tensor(param_data).reshape(value.size())
        # Set the parameter tensor in the model's state dict
        state_dict[key] = param_tensor
        # Update the start index for the next parameter
        start_index += size

    # Load the state dict into the model
    # model.load_state_dict(state_dict)
    return state_dict


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
    parser.add_argument('--data_dir', type=str, help="the data folder", dest="data_dir")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size: B")
    parser.add_argument("--num_clients", default=3, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=1234)

    # parameters required by the zkp library
    parser.add_argument("--max_malicious_clients", default=1, type=int, help="maximum number of malicious clients")
    parser.add_argument("--num_blinds_per_group_element", default=1, type=int, help="number of blinds per group element")
    parser.add_argument("--weight_bits", default=16, type=int, help="number of bits of weight updates")
    parser.add_argument("--random_normal_bit_shifter", default=24, type=int,
                        help="random normal samples are multiplied by 2^random_normal_bit_shifter and "
                             "rounded to the nearest integer. The paper uses 24.")
    parser.add_argument("--num_norm_bound_samples", default=1000, type=int,
                        help="number of multidimensional normal samples used in proving the l2 norm bound")
    parser.add_argument("--inner_prod_bound_bits", default=44, type=int,
                        help="the number of bits of each inner product between the model update and discretized "
                             "multidimensional normal sample. Recommended: weight_bits + random_normal_bit_shifter + 4")
    parser.add_argument("--max_bound_sq_bits", default=100, type=int,
                        help="the maximum number of bits of the sum of squares of inner products. "
                             "Recommended: 2 * (weight_bits + random_normal_bit_shifter) + 20")
    parser.add_argument("--check_type", default=0, type=int,
                        help="the type of the check method. Supported: l2 norm check 0, "
                             "sphere check 1, cosine similarity check 2")
    parser.add_argument("--norm_bound", default=2.0, type=float, help="the norm bound of each client's update")
    parser.add_argument("--b_precomp", default=False, type=bool, help="whether store the precomputed group elements")
    # TODO: this parameter shall be inferred based on the model selected, not given by argument
    parser.add_argument("--dim", default=4746, type=int, help="the dimension of the model")

    args = parser.parse_args(arg)
    return args
