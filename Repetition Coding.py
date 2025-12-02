# %%
import copy
import math
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
import flwr_datasets
from datasets import concatenate_datasets, load_from_disk
import pandas as pd


# %%
import argparse

parser = argparse.ArgumentParser(description="FL Simulation Script")
parser.add_argument('--ne', type=int, default=20, help='Number of clients')
parser.add_argument('--nh', type=int, default=5, help='Number of helpers')
parser.add_argument('--s', type=int, default=2, help='Number of erasures')
parser.add_argument('--r', type=int, default=5, help='Number of rounds')
parser.add_argument('--groupsize',type=int,default=10,help='groupsize')
args = parser.parse_args()

num_clients = args.ne
num_helpers = args.nh
num_erasures = num_helpers-1
NUM_ROUNDS = args.r
GROUP_SIZE = args.groupsize

# %%
# num_helpers = 5
# num_clients = 20
# num_erasures = num_helpers-1
# NUM_ROUNDS = 5

# %%
DATASET_PATH='femnist_data'
# GROUP_SIZE=10


# %%


# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
curr_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
import random, os
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)  # call before creating model, dataset, etc.
g = torch.Generator().manual_seed(42)

# %%
def generate_full_rank_matrix_with_identity(n, k, max_int=100):
    identity = [[int(i == j) for j in range(n)] for i in range(n)]
    
    while True:
        extra_rows = [[int(np.random.randint(0, max_int + 1)) for _ in range(n)] for _ in range(k)]
        candidate_matrix = identity + extra_rows
        if np.linalg.matrix_rank(np.array(candidate_matrix)) == n:
            return candidate_matrix

# %%
from collections import defaultdict
import random

def generate_failure_matrix(ne,nh,s):
    failure_matrix = [[0]*nh for _ in range(ne)]
    failure_dict = defaultdict(list)
    for i in range(ne):
        failed_connections = tuple(random.sample(range(0,nh),nh-s-1))
        failure_dict[failed_connections].append(i)
        for j in failed_connections:
            failure_matrix[i][j]=1
    return failure_matrix,dict(failure_dict)

# %%


# %%
GLOBAL_FAILURE_MATRIX = generate_failure_matrix(num_clients,num_helpers,num_erasures)[0]



# %%
def payload(item):
    if item is None:
        return 0
    ret = 0
    state = item[0]
    for k in state:
        if is_0d_tensor_pytorch(state[k]):
            ret += 4
        else:
            ret += state[k].nbytes
    return ret

def is_0d_tensor_pytorch(value):
    return torch.is_tensor(value) and value.dim() == 0


# %%

import random, os
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For reproducible dataloader workers
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(42)  
g = torch.Generator().manual_seed(42)


# %%
class HFWrapper(Dataset):
    def __init__(
        self, hf_dataset, image_key="image", label_key="character",
        # transform=transforms.ToTensor()
    ):
        self.hf = hf_dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transforms.ToTensor()
        
    def __len__(self):
        return len(self.hf)

    def __getitem__(self, idx):
        item = self.hf[idx]
        img = item[self.image_key]
        label = item[self.label_key]
        img = self.transform(img)
        return img, label




def make_loader(hf_dataset, batch_size=32, shuffle=False):
    wrapped = HFWrapper(hf_dataset)
    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=2,
        pin_memory=True,
    )

# %%
class FEMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 7 * 7, 62)   # 62 classes in FEMNIST

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)

# %%
class LogisticRegression(nn.Module):
    def __init__(self, input_dim=784, num_classes=62):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten 28x28
        return self.fc(x)

# %%
class Helper(dict):
    def __init__(self,idx):
        self.hid = idx


helper_nodes = []
for i in range(num_helpers):
    helper_nodes.append(Helper(i))

# %%
ctoh_payloads = {}
htom_payloads = defaultdict(int)

# %%
import random
def client_to_helper(cid,item):
    
    for i in range(num_helpers):
        helper_nodes[i][cid] = item
    for j in range(len(GLOBAL_FAILURE_MATRIX[cid])):
        if GLOBAL_FAILURE_MATRIX[cid][j] == 1:
            helper_nodes[j][cid]=None
  
def helper_to_master():
    updates = []
    updates = [None for _ in range(num_clients)]
    for h in helper_nodes:
        tot_pload = 0
        for entry in h.items():
            if entry[1] is not None and updates[entry[0]] is None :
                updates[entry[0]]=entry[1]
                tot_pload = payload(entry[1])
            
        htom_payloads[f'helper_{h.hid}'] += tot_pload
        print(f'payload of helperid {h.hid}: is {tot_pload}')
    return updates



# %%
class Client:
    def __init__(self,cid, dataset, device=curr_device):
        self.cid = cid
        self.dataset = dataset
        self.device = device
        self.batch_size = 32
        self.loader = make_loader(self.dataset, batch_size=self.batch_size, shuffle=True)


    def train_local(self, global_model, epochs=1, batch_size=32, lr=0.01):
        self.batch_size = batch_size
        model = copy.deepcopy(global_model).to(self.device)
        # loader = make_loader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for X, y in self.loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()

        client_to_helper(self.cid,(model.state_dict(), len(self.dataset)))
        return model.state_dict(), len(self.dataset)


# %%
class Master:
    def __init__(self, global_model, clients, device=curr_device):
        self.global_model = global_model
        self.clients = clients
        self.device = device

    def aggregate(self, updates):
        num_samples = [num for  _,num in updates]
        total_samples = sum(num_samples)
        
        scaling_factors = torch.tensor(num_samples,device=self.device,dtype=torch.float32)
        scaling_factors /= total_samples
        
        new_state = {}

        for key in updates[0][0].keys():
            
            param_stack = torch.stack([
                client_state[key].to(self.device) for client_state, _ in updates
            ])
            
            view_shape = [-1] + [1] * (param_stack.ndim - 1)
            scales = scaling_factors.view(*view_shape)
            
            weighted_sum = (param_stack * scales).sum(dim=0)
            
            
            new_state[key] = weighted_sum

        self.global_model.load_state_dict(new_state)

    def run_round(self, epochs=1, batch_size=32, lr=0.01):
        updates = [None for _ in range(num_clients)]
        for client in self.clients:
            client_state, num_samples = client.train_local(
                self.global_model, epochs=epochs, batch_size=batch_size, lr=lr
            )
        updates = helper_to_master()
        valid_updates = [u for u in updates if u is not None]
        if valid_updates:
            self.aggregate(updates)

    '''For validation use this run round function'''
    # def run_round(self, epochs=1, batch_size=32, lr=0.01):
    #     updates = [None for _ in range(num_clients)]
    #     for client in self.clients:
    #         client_state, num_samples = client.train_local(
    #             self.global_model, epochs=epochs, batch_size=batch_size, lr=lr
    #         )
    #         updates[client.cid] = (client_state,num_samples)
    #     valid_updates = [u for u in updates if u is not None]
    #     if valid_updates:
    #         self.aggregate(updates)



    def evaluate(self, test_dataset, batch_size=64):
        loader = make_loader(test_dataset, batch_size=batch_size)
        model = self.global_model.to(self.device)
        model.eval()

        total, correct, total_loss = 0, 0, 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device, non_blocking=True), y.to(self.device,non_blocking=True)
                outputs = model(X)
                total_loss += criterion(outputs, y).item() * y.size(0)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)

        return total_loss / total, correct / total

# %%
def weighted_average(updates):
    """Weighted average of model weights based on dataset sizes."""
    total_examples = sum(num for _, num in updates)
    new_state = {}

    for key in updates[0][0].keys():
        new_state[key] = sum((state[key] * num for state, num in updates), 
                             torch.zeros_like(updates[0][0][key]))
        new_state[key] /= total_examples

    return new_state, total_examples

# %%
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, GroupedNaturalIdPartitioner
from datasets import ClassLabel
from torch.utils.data import DataLoader




client_subsets = []
test_datasubsets = []





class OptimizedFemnistLoader:
    def __init__(self, path, group_size):
        print(f"Loading dataset from: {path}")
        self.data = load_from_disk(path)["train"]
        self.group_size = group_size
        
        # --- OPTIMIZATION START ---
        # Instead of filtering later, we map writers to indices NOW using Pandas (very fast)
        print("Indexing data by writer_id...")
        
        # 1. Pull just the writer_id column (cheap operation)
        writer_ids = self.data['writer_id']
        
        # 2. Use Pandas to group indices by writer_id
        df = pd.DataFrame({'writer_id': writer_ids, 'index': range(len(writer_ids))})
        # This creates a dictionary: {writer_id: [index, index, ...]}
        self.writer_map = df.groupby('writer_id')['index'].apply(list).to_dict()
        
        # 3. Get sorted unique writers
        self.unique_writers = sorted(self.writer_map.keys())
        print(f"Indexing complete. Found {len(self.unique_writers)} writers.")
        # --- OPTIMIZATION END ---

    def get_client_dataset(self, client_id):
        # 1. Calculate which writers belong to this client
        start_idx = client_id * self.group_size
        end_idx = start_idx + self.group_size
        
        if start_idx >= len(self.unique_writers):
            raise IndexError("Client ID out of range for available writers")
            
        target_writers = self.unique_writers[start_idx:end_idx]
        
        # 2. Collect all indices for these writers
        # This is instantaneous, no searching required
        indices_to_select = []
        for writer in target_writers:
            indices_to_select.extend(self.writer_map[writer])
            
        # 3. Use .select() instead of .filter()
        # .select() grabs rows by index instantly without scanning the whole file
        return self.data.select(indices_to_select)
    
loader = OptimizedFemnistLoader(DATASET_PATH,GROUP_SIZE)




for i in range(num_clients):
    curr_part = loader.get_client_dataset(i)
    split_data = curr_part.train_test_split(test_size=0.2, seed=42)    
    assert len(split_data['test'])>0 and len(split_data['train'])>0
    client_subsets.append(split_data["train"])
    test_datasubsets.append(split_data["test"])
clients = [Client(cid=i, dataset=client_subsets[i],device=curr_device) for i in range(num_clients)]


train_data = concatenate_datasets(client_subsets)
train_loader = make_loader(train_data, batch_size=256, shuffle=True)
test_data = concatenate_datasets(test_datasubsets)
test_loader = make_loader(test_data,batch_size=256,shuffle=False)





# %%
cl_acc = []
def train_central(model, train_loader, test_loader, device=curr_device, epochs=NUM_ROUNDS):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}: Test Accuracy: {acc*100:.2f}%")
        cl_acc.append(acc*100)



# %%

global_model = FEMNISTNet().to(curr_device)
# global_model = LogisticRegression().to(curr_device)
master = Master(global_model, clients, device=curr_device)

fl_acc = []
fl_loss = []
for r in range(1, NUM_ROUNDS + 1):
    print(f"--- Round {r} ---")
    GLOBAL_FAILURE_MATRIX = generate_failure_matrix(num_clients,num_helpers,num_erasures)[0]
    print(GLOBAL_FAILURE_MATRIX)
    master.run_round(epochs=1, batch_size=32, lr=0.05)
    loss, acc = master.evaluate(test_data)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc*100:.2f}%")
    fl_acc.append(acc)
    fl_loss.append(loss)

# %%
print(fl_acc)

# %%
print(fl_loss)

# %%
print(htom_payloads)

# %%
import pickle, fcntl, os
def save_exp_res(filename,args,results):
    args_keys = tuple(args)
    if not os.path.exists(filename):
        with open(filename, 'wb') as f:
            pickle.dump({}, f)
    with open(filename, 'rb+') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:
            try:
                data = pickle.load(f)
            except EOFError:
                data = {}
            data[args_keys] = results
            f.seek(0)
            f.truncate()
            pickle.dump(data,f)
        finally:
            fcntl.flock(f,fcntl.LOCK_UN)

#%%

save_exp_res('cnn_arc.pkl',(num_clients,num_helpers,num_erasures,NUM_ROUNDS,GROUP_SIZE),fl_acc)

#%%
save_exp_res('cnn_payload_arc.pkl',(num_clients,num_helpers,num_erasures,NUM_ROUNDS,GROUP_SIZE),dict(htom_payloads))
