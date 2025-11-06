import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from TrustGraph import build_trust_graph_from_json_with_llm
from Code.EN.CounterFactual import build_counter_trust_graph_from_json_with_llm


DATA_FILE = ""
# DATA_FILE = ""
TRUST_CACHE = ""
COUNTER_CACHE = ""
LLM_OUTPUT_FILE = ""
LLM_counter_OUTPUT_FILE = ""
GRAPH_DIM = 
SIS_DIM = 
BATCH_SIZE = 


with open(DATA_FILE, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
df_labels = pd.DataFrame([
    {"news_id": k, "label": v["label"]}
    for k, v in raw_data.items()
])
df_labels["news_id"] = df_labels["news_id"].astype(str)



orig_graphs, _ = build_trust_graph_from_json_with_llm(DATA_FILE, TRUST_CACHE, LLM_OUTPUT_FILE)

cf_graphs, _ = build_counter_trust_graph_from_json_with_llm(DATA_FILE, COUNTER_CACHE, LLM_counter_OUTPUT_FILE)


def graph_to_vector(graph: nx.DiGraph, dim=GRAPH_DIM):
    deg = np.array([d for _, d in graph.degree()])
    if len(deg) == 0:
        return np.zeros(dim)
    vec = np.pad(deg.mean().reshape(1), (0, dim - 1))[:dim]
    return vec.astype(np.float32)

def compute_sis(g1: nx.DiGraph, g2: nx.DiGraph) -> float:
    nodes = sorted(set(g1.nodes()) | set(g2.nodes()))
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    for u, v, d in g1.edges(data=True):
        A[idx[u], idx[v]] = d.get("weight", 0)
    for u, v, d in g2.edges(data=True):
        B[idx[u], idx[v]] = d.get("weight", 0)
    return np.linalg.norm(A - B)


records = []
for news_id in orig_graphs:
    news_id_str = str(news_id)
    if news_id_str not in cf_graphs:
        continue
    g1 = orig_graphs[news_id_str]
    g2 = cf_graphs[news_id_str]
    sis = compute_sis(g1, g2)
    vec1 = graph_to_vector(g1)
    vec2 = graph_to_vector(g2)
    records.append({
        "news_id": news_id_str,
        "SIS": sis,
        "g1": vec1,
        "g2": vec2
    })


df_feat = pd.DataFrame(records)
df_all = pd.merge(df_feat, df_labels, on="news_id")

# ==== Dataset ====
class FusionDataset(Dataset):
    def __init__(self, df):
        self.g1 = torch.tensor(np.stack(df["g1"]), dtype=torch.float32)
        self.g2 = torch.tensor(np.stack(df["g2"]), dtype=torch.float32)
        self.sis = torch.tensor(df["SIS"].values, dtype=torch.float32).unsqueeze(1)
        self.label = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.g1[idx], self.g2[idx], self.sis[idx], self.label[idx]

train_df, temp_df = train_test_split(df_all, test_size=0.3, stratify=df_all["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

train_loader = DataLoader(FusionDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(FusionDataset(val_df), batch_size=BATCH_SIZE)
test_loader = DataLoader(FusionDataset(test_df), batch_size=BATCH_SIZE)


class RumorClassifier(nn.Module):
    def __init__(self, g_dim=GRAPH_DIM, sis_dim=SIS_DIM, hidden=128):
        super().__init__()
        self.sis_embed = nn.Sequential(
            nn.Linear(1, sis_dim),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(2 * g_dim + sis_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )

    def forward(self, g1, g2, sis):
        sis_vector = self.sis_embed(sis)
        x = torch.cat([g1, g2, sis_vector], dim=1)
        return self.classifier(x).squeeze(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RumorClassifier().to(device)
EPOCHS = 100
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

def evaluate(loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for g1, g2, sis, y in loader:
            g1, g2, sis, y = g1.to(device), g2.to(device), sis.to(device), y.to(device)
            out = model(g1, g2, sis)
            pred = (torch.sigmoid(out) > 0.5).long()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return classification_report(y_true, y_pred, output_dict=False)

for epoch in range(1, EPOCHS + 1):
    model.train()
    for g1, g2, sis, y in train_loader:
        g1, g2, sis, y = g1.to(device), g2.to(device), sis.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(g1, g2, sis)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    print(f"\n🎯 Epoch {epoch} completed. Validation Result:")
    print(evaluate(val_loader))


print("\n✅ Final Test Result:")
print(evaluate(test_loader))
