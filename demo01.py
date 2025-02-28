import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import pandas as pd
from typing import Union
from tqdm import tqdm
import time


class Transformer(nn.Module):
    def __init__(self, d_model, d_k, d_v, head, d_ff, N, dropout):
        super(Transformer, self).__init__()
        self.encoder = nn.ModuleList([EncoderBlock(d_model, d_k, d_v, head, d_ff, dropout) for _ in range(N)])
    
    def forward(self, x, mask=None):
        for encoder in self.encoder:
            x = encoder(x, mask)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, head, d_ff, dropout):
        super(EncoderBlock, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, d_k, d_v, head)
        self.sublayer_1 = SubLayer(self.multihead_attention, dropout)
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.sublayer_2 = SubLayer(self.feed_forward, dropout)
        self.layernorm_2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        x = self.sublayer_1(x, mask)
        x = self.layernorm_1(x)
        x = self.sublayer_2(x)
        x = self.layernorm_2(x)
        return x


class SubLayer(nn.Module):
    def __init__(self, layer, dropout):
        super(SubLayer, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.dropout(self.layer(x, mask))
        else:
            x = x + self.dropout(self.layer(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, head):
        super(MultiHeadAttention, self).__init__()
        assert d_k % head == 0 and d_v % head == 0
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.head = head
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)
        self.W_O = nn.Linear(d_v, d_model)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.W_Q(x).view(batch_size, -1, self.head, self.d_k // self.head).transpose(1, 2)
        K = self.W_K(x).view(batch_size, -1, self.head, self.d_k // self.head).transpose(1, 2)
        V = self.W_V(x).view(batch_size, -1, self.head, self.d_v // self.head).transpose(1, 2)
        result = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, head, seq_len, seq_len)
        if mask is not None:  # (batch_size, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            result = result.masked_fill(mask == 1, -1e9)
        result = F.softmax(result, dim=-1)
        result = torch.matmul(result, V)
        result = result.transpose(1, 2).contiguous().view(batch_size, -1, self.d_v)
        result = self.W_O(result)
        return result


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.W_1 = nn.Linear(d_model, d_ff)
        self.W_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        result = F.relu(self.W_1(x))
        result = self.W_2(result)
        return result


class Embedding(nn.Module):
    def __init__(self, k, d_model):
        super(Embedding, self).__init__()
        self.E = nn.Parameter(torch.randn(k, d_model) * 0.05)
    
    @property
    def weight(self):
        return self.E
    
    def forward(self, idx):
        return self.weight[idx.to(torch.int), :]
    
    def __len__(self):
        return self.E.size(0)


class OrderedEmbedding(nn.Module):
    def __init__(self, occ, d_model):
        super(OrderedEmbedding, self).__init__()
        self.E = nn.Parameter(torch.zeros(len(occ), d_model))
        self.l = nn.Parameter((torch.randn(d_model) * 0.05).unsqueeze(0), requires_grad=False)
        self.h = nn.Parameter((torch.randn(d_model) * 0.05).unsqueeze(0), requires_grad=False)
        self.r = nn.Parameter(((occ - occ[0]) / (occ[-1] - occ[0])).unsqueeze(1), requires_grad=False)
    
    @property
    def weight(self):
        return self.r * self.l + (1 - self.r) * self.h + self.E
    
    def forward(self, idx):
        return self.weight[idx.to(torch.int), :]
    
    def __len__(self):
        return self.E.size(0)


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, dropout):
        super(PositionalEmbedding, self).__init__()
        self.E = nn.Parameter(torch.randn(seq_len, d_model) * 0.01)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x + self.E)


class DynamicLinear(nn.Module):
    def __init__(self, embedding : Union[Embedding, OrderedEmbedding]):
        super(DynamicLinear, self).__init__()
        self.E = embedding
        self.temp = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(len(embedding)))
    
    def forward(self, x):
        raw_logits = torch.matmul(x, self.E.weight.T) + self.bias
        return raw_logits / F.sigmoid(self.temp)


def kmeans(x, k, max_iter=100):
    centroids = x[torch.randperm(x.size(0))[:k]]
    
    for _ in tqdm(range(max_iter), "Kmeans processing"):
        distances = torch.abs(x.unsqueeze(1) - centroids)
        cluster_assignments = torch.argmin(distances, dim=1)
        new_centroids = torch.stack([x[cluster_assignments == i].mean() if (cluster_assignments == i).sum() > 0 else centroids[i] for i in range(k)])
        if torch.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    centroids, sorted_indices = torch.sort(centroids)
    cluster_assignments = torch.gather(sorted_indices, 0, cluster_assignments)

    return centroids, cluster_assignments


class TabMT(nn.Module):
    def __init__(self, d_model, d_k, d_v, head, d_ff, N, dropout, info):
        super(TabMT, self).__init__()
        
        self.schema = info["schema"]
        self.columns = info["columns"]
        self.cate2int = info["cate2int"]
        self.int2cate = info["int2cate"]
        self.seq_len = info["seq_len"]
        self.occs = info["occs"]
        self.ks = info["ks"]

        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.05, requires_grad=False)

        self.embeddings = nn.ModuleList([Embedding(self.ks[i], d_model) if schema[i] == 'c' else OrderedEmbedding(self.occs[i], d_model) for i in range(seq_len)])
        self.positional_embedding = PositionalEmbedding(seq_len, d_model, dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.transformer = Transformer(d_model, d_k, d_v, head, d_ff, N, dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dynamic_linears = nn.ModuleList([DynamicLinear(e) for e in self.embeddings])
    
    def forward(self, x : Union[pd.DataFrame, torch.Tensor], mask=None):        
        x_embed = []
        for i in range(self.seq_len):
            col = self.embeddings[i](x[:, i])
            x_embed.append(col)
        x = torch.stack(x_embed, dim=1)
        
        x[mask] = self.mask_token

        x = self.positional_embedding(x)
        x = self.dropout_1(x)

        x = self.transformer(x, mask)
        x = self.dropout_2(x)

        preds = []
        for i in range(self.seq_len):
            col = self.dynamic_linears[i](x[:, i, :])
            preds.append(col)
        
        return preds

    def encode(self, x : pd.DataFrame):
        x.columns = columns
        for i in range(len(self.columns)):
            if self.schema[i] == 'c':
                col = x.columns[i]
                x[col] = x[col].map(self.cate2int[col])
        
        x = torch.tensor(x.to_numpy(), dtype=torch.float32)

        for i in range(self.seq_len):
            if schema[i] == 'n':
                diff = torch.abs(x[:, i].unsqueeze(1) - self.occs[i])
                _, indices = torch.min(diff, dim=1)
                x[:, i] = indices
        
        return x
    
    def decode(self, x : torch.Tensor):
        x = x.detach().clone().cpu()
        for i in range(self.seq_len):
            if schema[i] == 'n':
                x[:, i] = self.occs[i][x[:, i]]

        x = pd.DataFrame(x.numpy(), columns=self.columns)

        for i in range(self.seq_len):
            if self.schema[i] == 'c':
                col = x.columns[i]
                x[col] = x[col].map(self.int2cate[col])
        
        return x


class MyDataset(Dataset):
    def __init__(self, data : torch.Tensor, tgt=None):
        self.data = data
        self.tgt = tgt
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.tgt is not None:
            return self.data[idx], self.tgt[idx]
        else:
            return self.data[idx]


def save(model, optimizer, epoch, path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "info": {
            "columns": model.columns,
            "schema": model.schema,
            "cate2int": model.cate2int,
            "int2cate": model.int2cate,
            "seq_len": model.seq_len,
            "occs": model.occs,
            "ks": model.ks
        }

    }
    torch.save(checkpoint, path)


def load(path):
    checkpoint = torch.load(path)
    return checkpoint


def get_data_info(data : pd.DataFrame, columns, schema, kmeans_cluster_nums, kmeans_max_iters):
    info = {}
    data.columns = columns
    info["columns"] = columns
    info["schema"] = schema
    ks = [0] * len(schema)
    cate2int = {}
    int2cate = {}
    for i in range(len(columns)):
        if schema[i] == 'c':
            print(f"build cate2int for column {columns[i]}")
            col = data.columns[i]
            ks[i] = len(data[col].unique())
            cate2int[col] = {colv: idx for idx, colv in enumerate(data[col].unique().tolist())}
            int2cate[col] = {idx: colv for idx, colv in enumerate(data[col].unique().tolist())}
            data[col] = data[col].map(cate2int[col])
    info["cate2int"] = cate2int
    info["int2cate"] = int2cate
    data = torch.tensor(data.to_numpy(), dtype=torch.float32)
    seq_len = data.size(1)
    info["seq_len"] = seq_len
    occs = []
    for i in range(seq_len):
        if schema[i] == 'c':
            occs.append([])
        elif schema[i] == 'n':
            print(f"Kmeans on column {columns[i]}")
            occ, _ = kmeans(data[:, i], kmeans_cluster_nums[i], max_iter=kmeans_max_iters[i])
            occs.append(occ)
            ks[i] = kmeans_cluster_nums[i]
    info["occs"] = occs
    info["ks"] = ks

    return info


if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print("device: ", device)

    print("load data")
    x = pd.read_csv("/home/yc/dataset/fflights/fflights.csv")
    print("load data success")
    columns = x.columns.tolist()
    schema = ['c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'n', 'n', 'n', 'n', 'n', 'n']
    seq_len = 15
    
    kmeans_cluster_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 64, 64, 64, 64, 64]
    kmeans_max_iters = [0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 100, 100]

    d_model = d_k = d_v = 512
    head = 8
    d_ff = 2048
    N = 12
    dropout = 0.1

    epochs = 0
    batch_size = 2048

    save_model = True
    save_model_epoch = 10
    save_dir = "./models/v2/"
    load_model = False
    load_model_path = "./models/model_ep50.pth"

    eval = False

    if load_model:
        checkpoint = load(load_model_path)
        info = checkpoint['info']
        print(f"load model [epoch: {checkpoint['epoch']}]")
    else:
        print("get data info ...")
        info = get_data_info(x.copy(), columns, schema, kmeans_cluster_nums, kmeans_max_iters)
        print("get data info finished")

    model = TabMT(d_model, d_k, d_v, head, d_ff, N, dropout, info).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()

    if load_model:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        save(model, optimizer, 0, f"{save_dir}model_ep0.pth")
        print(f"save model [epoch 0] to {save_dir}model_ep0.pth")

    print("encode data from dataframe to tensor")
    data = model.encode(x.copy())
    label = model.encode(x.copy()).to(torch.int64)

    dataset = MyDataset(data, label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("start training")
    model.train()
    start_epoch = checkpoint['epoch'] + 1 if load_model else 1
    for epoch in range(start_epoch, epochs + 1):
        total_loss = 0
        mask_count = 0
        start_time = time.perf_counter()
        # tqdm_bar = tqdm(enumerate(dataloader), f"Training Epoch {epoch}", total=len(dataloader))
        for batch_idx, (batch, tgt) in enumerate(dataloader):
            mask = torch.randn_like(batch) > torch.randn(batch.size(0), 1)
            while mask.sum() == 0:
                mask = torch.randn_like(batch) > torch.randn(batch.size(0), 1)
            
            batch = batch.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)

            preds = model(batch, mask)
    
            loss = 0.0
            for i in range(seq_len):
                if mask[:, i].sum() == 0:
                    continue
                loss += loss_fn(preds[i][mask[:, i]], tgt[:, i][mask[:, i]])
            loss /= seq_len
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            # tqdm_bar.set_postfix(loss=f"{loss.item():.8f}")
            total_loss += loss.item() * mask.sum().item()
            mask_count += mask.sum().item()
        
        end_time = time.perf_counter()

        print('Epoch: {}, Total Loss: {}, Average Loss: {}, Time: {}'.format(epoch, total_loss, total_loss / mask_count, end_time - start_time))

        if save_model and epoch % save_model_epoch == 0:
            save(model, optimizer, epoch, f"{save_dir}model_ep{epoch}.pth")
            print(f"save model [epoch {epoch}] to {save_dir}model_ep{epoch}.pth")
        
    if eval:
        model.eval()
        gen_num = 1000
    
        batch = torch.zeros(gen_num, seq_len)
        mask = torch.ones_like(batch, dtype=torch.bool)
    
        # batch = model.encode(x.sample(gen_num))
        # mask = (torch.randn_like(batch) > torch.randn(batch.size(0), 1))
        # mask[:, -1] = 1
    
        temps = torch.tensor([1] * seq_len)
    
        batch = batch.to(torch.int64).to(device)
        mask = mask.to(device)
        temps = temps.to(device)
    
        for i in torch.randperm(seq_len):
            preds = model(batch, mask)
            batch[:, i] = torch.multinomial(F.softmax(preds[i] / temps, dim=1), 1).squeeze()
            # batch[mask[:, i], i] = torch.argmax(F.softmax(preds[i] / temp, dim=1), 1)[mask[:, i]]
            mask[:, i] = False
        
        gen_data = model.decode(batch)
    
        print(x[x['Quarter'] == 1]['DepDelay'].mean())
        print(gen_data[gen_data['Quarter'] == 1]['DepDelay'].mean())
    
        print(x[x['Month'] == 5]['TaxiOut'].mean())
        print(gen_data[gen_data['Month'] == 5]['TaxiOut'].mean())
    
        print(x[x['Reporting_Airline'] == 'YX']['ArrDelay'].mean())
        print(gen_data[gen_data['Reporting_Airline'] == 'YX']['ArrDelay'].mean())
    
        print(x[x['Origin'] == 'CMH']['TaxiIn'].mean())
        print(gen_data[gen_data['Origin'] == 'CMH']['TaxiIn'].mean())
    
        print(x[x['Dest'] == 'DCA']['AirTime'].mean())
        print(gen_data[gen_data['Dest'] == 'DCA']['AirTime'].mean())
    
        print(x[x['DestStateName'] == 'Virginia']['Distance'].mean())
        print(gen_data[gen_data['DestStateName'] == 'Virginia']['Distance'].mean())
