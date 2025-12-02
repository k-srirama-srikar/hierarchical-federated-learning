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
# import flwr_datasets
from datasets import concatenate_datasets, load_from_disk
from collections import OrderedDict
import pandas as pd

import time
import argparse

start = time.time()


#%%

parser = argparse.ArgumentParser(description="FL Simulation Script")
parser.add_argument('--ne', type=int, default=20, help='Number of clients')
parser.add_argument('--nh', type=int, default=5, help='Number of helpers')
parser.add_argument('--s', type=int, default=2, help='Number of erasures')
parser.add_argument('--r', type=int, default=5, help='Number of rounds')
parser.add_argument('--groupsize',type=int,default=10,help='groupsize')
args = parser.parse_args()

#%%
curr_dev = 'cuda'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
from collections import defaultdict

# %%
def generate_failure_matrix(ne,nh,s):
    failure_matrix = [[0]*nh for _ in range(ne)]
    failure_dict = defaultdict(list)
    for i in range(ne):
        failed_connections = tuple(random.sample(range(0,nh),s))
        failure_dict[failed_connections].append(i)
        for j in failed_connections:
            failure_matrix[i][j]=1
    return failure_matrix,dict(failure_dict)

# %%
def generate_full_rank_matrix_with_identity(n, k, max_int=100):
    identity = [[int(i == j) for j in range(n)] for i in range(n)]
    
    while True:
        extra_rows = [[int(np.random.randint(0, max_int + 1)) for _ in range(n)] for _ in range(k)]
        candidate_matrix = identity + extra_rows
        if np.linalg.matrix_rank(np.array(candidate_matrix)) == n:
            return candidate_matrix

# %%
# num_helpers = 5
# num_clients = 20
# num_erasures = 2
# NUM_ROUNDS = 5

num_clients = args.ne
num_helpers = args.nh
num_erasures = args.s
NUM_ROUNDS = args.r
GROUP_SIZE = args.groupsize


GLOBAL_FAILURE_MATRIX,GLOBAL_FAILURE_DICT = generate_failure_matrix(num_clients,num_helpers,num_erasures)
# MDS_CODE = [[1,0,0],[0,1,0],[0,0,1],[1,1,1],[1,2,3]]
MDS_CODE = torch.tensor(generate_full_rank_matrix_with_identity(num_helpers-num_erasures,num_erasures)).to(device)
MDS_CODE = torch.tensor(MDS_CODE, dtype=torch.float32)
# MDS_CODE_NP = np.array(MDS_CODE)

# %%
print(MDS_CODE)

# %%


# %%
# GLOBAL_FAILURE_MATRIX

# %%
GLOBAL_FAILURE_MATRIX

# %%
# GLOBAL_FAILURE_DICT

# %%
print(GLOBAL_FAILURE_DICT.keys())

# %%
from collections import defaultdict
payload_dict = defaultdict(list)

#%%
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
model = FEMNISTNet()
for name, param in model.state_dict().items():
    print(name, param.shape,param.numel())

# %%
metadata_of_logistic_regression = {
    ('fc.weight', (62, 784), 62*784),
    ('fc.bias', (62,), 62)
}


# %%
metadata_of_femnistnet = [
    ('conv1.weight', (32, 1, 7, 7), 1568),
    ('conv1.bias', (32,), 32),
    ('conv2.weight', (64, 32, 3, 3), 18432),
    ('conv2.bias', (64,), 64),
    ('fc.weight', (62, 3136), 194432),
    ('fc.bias', (62,), 62),
]

# %%
def flatten_state_dict(state_dict,metadata=metadata_of_femnistnet):
    flat_list = []
    for key, shape, numel in metadata:
        flat_list.append(state_dict[key].view(-1))
    return torch.cat(flat_list)

def split_tensor(tensor, num_parts):
    n = tensor.size(0)
    tensor = torch.cat([tensor, torch.zeros(num_parts - n % num_parts, dtype=tensor.dtype, device=tensor.device)])
    n = tensor.size(0)
    split_tensors = torch.split(tensor, n // num_parts)
    return torch.stack(split_tensors)

def rebuild_state_dict_from_flat(flat_list,metadata=metadata_of_femnistnet):
    new_state_dict = OrderedDict()
    count = 0
    for key, shape, numel in metadata:
        new_state_dict[key] = flat_list[count:count+numel].view(shape)
        count += numel
    return new_state_dict

def rebuild_flat_tensor_from_pieces(tensor_pieces,len_tensor=214590):
    fin_piece = tensor_pieces.reshape(-1)
    fin_piece = fin_piece[:len_tensor]
    return fin_piece

def payload(tensor):
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()
    # return tensor.nbytes


# %%
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, GroupedNaturalIdPartitioner
from datasets import ClassLabel
from torch.utils.data import DataLoader

class HFWrapper(Dataset):
    def __init__(
        self, hf_dataset, image_key="image", label_key="character",
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
        num_workers=4,
        pin_memory=True)


DATASET_PATH = 'femnist_data'




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



client_subsets = []
test_datasubsets = []
for i in range(num_clients):
    # curr_part = dataset.load_partition(partition_id=i, split="train")
    curr_part = loader.get_client_dataset(i)
    split_data = curr_part.train_test_split(test_size=0.2, seed=42)    
    assert len(split_data['test'])>0 and len(split_data['train'])>0
    client_subsets.append(split_data["train"])
    test_datasubsets.append(split_data["test"])


train_data = concatenate_datasets(client_subsets)
train_loader = make_loader(train_data, batch_size=256, shuffle=True)
test_data = concatenate_datasets(test_datasubsets)
test_loader = make_loader(test_data,batch_size=256,shuffle=False)


# %%
for i, train_subset in enumerate(client_subsets):
    print(f"Client {i} train subset size: {len(train_subset)}")

tot_size = 0
for i, train_subset in enumerate(client_subsets):
    tot_size += len(train_subset)

# %%
class Helper(OrderedDict):
    def __init__(self,hid):
        super().__init__()
        self.hid = hid
    def send_model_to_master(self):
        pass

# %%
helpers = [Helper(hid=i) for i in range(num_helpers)]

# %%
class Client:
    def __init__(self,cid, dataset, device=curr_dev):
        self.cid = cid
        self.dataset = dataset
        self.device = device
        self.model_state = None
        self.len_dataset = len(self.dataset)
        self.split_weights = None

    def train_local(self, global_model, epochs=1, batch_size=32, lr=0.01):
        model = copy.deepcopy(global_model).to(self.device)
        loader = make_loader(self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for X, y in loader:
                X, y = X.to(self.device,non_blocking=True), y.to(self.device,non_blocking=True)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
        self.model_state = copy.deepcopy(model.state_dict())
    
    def get_flatten_state_dict(self):
        weights = flatten_state_dict(self.model_state.copy())
        fed_wt = self.len_dataset / tot_size
        return fed_wt*weights
    
    def send_model_to_helper(self):
        weights = flatten_state_dict(self.model_state.copy())
        if weights.device != MDS_CODE.device:
            weights = weights.to(MDS_CODE.device)
        fed_wt = self.len_dataset / tot_size
        weights = weights * fed_wt
        self.true_weights = weights
        self.split_weights = split_tensor(weights, num_helpers-num_erasures)
        self.split_weights = MDS_CODE @ self.split_weights
        for i in range(num_helpers):
            helpers[i][self.cid] = self.split_weights[i]
            
        for i in range(len(GLOBAL_FAILURE_MATRIX[self.cid])):
            if GLOBAL_FAILURE_MATRIX[self.cid][i]==1:
                helpers[i][self.cid] = None
        

# %%
clients = [Client(cid=i, dataset=client_subsets[i]) for i in range(num_clients)]


# %%
class Master:
    def __init__(self, global_model,device=curr_dev):
        set_seed(42)
        self.global_model = global_model
        self.device = device
    
    def aggregate(self):
        pass
    
    def send_model_to_clients(self):
        state = self.global_model.state_dict()
        for  client in clients:
            client.model_state = copy.deepcopy(state)

    def get_model_from_helpers(self):
        final_model = []
        for (failed_key,failed_conns) in GLOBAL_FAILURE_DICT.items():
            true_key = list(range(num_helpers))
            true_key: List
            for val in failed_key:
                true_key.remove(val)
            print(true_key)
            
            helpers_curr = list(map(lambda x: helpers[x],true_key))
            fin = []
            for h in helpers_curr:
                client_tensors = [h[conn] for conn in failed_conns]
                tensor_sum = torch.stack(client_tensors).sum(dim=0)
                payload_dict[f"helper_{h.hid}"].append(payload(tensor_sum))
                fin.append(tensor_sum)
            fin_tensor = torch.stack(fin)
            fail_sub = MDS_CODE[torch.tensor(true_key, device=self.device), :len(true_key)]
            fin_mat = torch.linalg.solve(fail_sub,fin_tensor)
            fin_ten = rebuild_flat_tensor_from_pieces(fin_mat)
            print("Shape obtained after linalg solve,",fin_ten.shape)

            final_model.append(fin_ten)
        
        tot_ten = torch.stack(final_model).sum(dim=0)
        
        return tot_ten
    
    def run_round(self,  epochs=1, batch_size=32, lr=0.01):
        for cli in clients:
            cli.train_local(self.global_model,epochs=epochs,batch_size=batch_size,lr=lr)
            cli.send_model_to_helper()
        final_model = self.get_model_from_helpers()
        fin_state_dict = rebuild_state_dict_from_flat(final_model)
        self.global_model.load_state_dict(fin_state_dict)
        
    '''Use this run round for validation'''
    # def run_round(self,  epochs=1, batch_size=32, lr=0.01):
    #     cli_state = []
    #     for cli in clients:
    #         cli.train_local(self.global_model,epochs=epochs,batch_size=batch_size,lr=lr)
    #         cli_state.append(cli.get_flatten_state_dict())
    #     cli_state = torch.stack(cli_state)
        
    #     final_model = cli_state.sum(dim=0)
    #     fin_state_dict = rebuild_state_dict_from_flat(final_model)
    #     self.global_model.load_state_dict(fin_state_dict)
            
            
    
    def evaluate(self, test_dataset, batch_size=32):
        loader = make_loader(test_dataset, batch_size=batch_size)
        model = self.global_model.to(self.device)
        model.eval()

        total, correct, total_loss = 0, 0, 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = model(X)
                total_loss += criterion(outputs, y).item() * y.size(0)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)
        loss = total_loss / total
        accuracy = correct / total
        return loss, accuracy

# %%


# %%

fl_acc = []
fl_loss = []

# %%
global_model = FEMNISTNet()
master = Master(global_model=global_model)
for round in range(1,NUM_ROUNDS+1):
    print(f"--- Round {round} ---")
    
        
    GLOBAL_FAILURE_MATRIX,GLOBAL_FAILURE_DICT = generate_failure_matrix(num_clients,num_helpers,num_erasures)
    MDS_CODE = torch.tensor(generate_full_rank_matrix_with_identity(num_helpers-num_erasures,num_erasures)).to(device)
    MDS_CODE = torch.tensor(MDS_CODE, dtype=torch.float32)
    
    
    
    master.run_round(epochs=2, batch_size=32, lr=0.05)
    loss, acc = master.evaluate(test_data)
    master.send_model_to_clients()
    fl_acc.append(acc)
    fl_loss.append(loss)


# %%
print(fl_loss,fl_acc)

# %%
payload_dict

# %%
for key in payload_dict.keys():
    payload_dict[key] = sum(payload_dict[key])

# %%
payload_dict

# %%
for i in range(num_helpers):
    if f'helper_{i}' not in payload_dict.keys():
        payload_dict[i]=0

# %%
csv_file=f'payloads_{num_helpers}_{num_erasures}_{num_clients}_lr.csv'

# %%
import pandas as pd

# %%
import os
if not os.path.isfile(csv_file):
    df = pd.DataFrame(columns=payload_dict.keys())
    df.to_csv(csv_file,index=False)

# %%
entry = pd.DataFrame([dict(payload_dict)])
entry.to_csv(csv_file,mode='a',header=False,index=False)

# %%
import pandas as pd

end = time.time()
print("Time taken on cpu:",start-end,'seconds')

#%%

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

save_exp_res('cnn_amc.pkl',(num_clients,num_helpers,num_erasures,NUM_ROUNDS,GROUP_SIZE),fl_acc)

#%%
save_exp_res('cnn_payload_amc.pkl',(num_clients,num_helpers,num_erasures,NUM_ROUNDS,GROUP_SIZE),dict(payload_dict))



