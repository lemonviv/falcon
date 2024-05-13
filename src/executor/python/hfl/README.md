# Horizontal Federated Learning

This is an example of horizontal federated learning (HFL), In HFL, there is a server and a set of clients. Each client has a local dataset.
In each iteration, each client trains the model using its local dataset and uploads the model gradient to the server, which aggregates to get the global
gradient using the Federated Average algorithm. The server sends the global gradient to all clients for iterative model training.
Here gives an example that uses the Bank dataset and an MLP model in HFL.

## Preparation

Install the requirements 

```bash
pip install -r requirements.txt
```

Download the bank dataset and split it into 3 partitions.

```bash
# 1. download the data from https://archive.ics.uci.edu/ml/datasets/bank+marketing
# 2. put it under the /data folder
# 3. run the following command which:
#    (1) splits the dataset into N subsets
#    (2) splits each subsets into train set and test set (8:2)
#    (3) puts the split sub-datasets to /FALCON_PATH/src/executor/python/data/bank/
python3 -m bank N
```

## Run the example

Run the server first (set the number of epochs to 3)

```bash
python3 -m src.server_zkp -m 3 --num_clients 3
```

Then, start 3 clients in different terminal

```bash
python3 -m src.client_zkp --model mlp --data bank --data_dir /opt/falcon/src/executor/python/data/bank -m 3 --num_clients 3 -i 0
python3 -m src.client_zkp --model mlp --data bank --data_dir /opt/falcon/src/executor/python/data/bank -m 3 --num_clients 3 -i 1
python3 -m src.client_zkp --model mlp --data bank --data_dir /opt/falcon/src/executor/python/data/bank -m 3 --num_clients 3 -i 2
```

Finally, the server and clients finish the FL training. 