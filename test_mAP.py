import argparse
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
from PIL import Image

from external.adaptors.fastreid_adaptor import FastReID

class ReIDListDataset(Dataset):
    def __init__(self, root_dir, list_path, transform=None, relabel=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(list_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if 3 == len(parts):
                    img_path, pid, cam_id = parts
                else:
                    raise ValueError(f"Invalid line format in {list_path}: {line}")
                
                self.samples.append((img_path, int(pid), int(cam_id)))

        # Map ids to class indices starting from 0
        if relabel:
            label_map = {pid: idx for idx, pid in enumerate(sorted(set(pid for _, pid, _ in self.samples)))}
            self.relabel(label_map)

    def relabel(self, label_map):
        self.label_map = label_map
        new_samples = []
        for img_path, pid, cam_id in self.samples:
            if pid in self.label_map:
                new_label = self.label_map[pid]
                new_samples.append((img_path, new_label, cam_id))
            else:
                print(f"PID {pid} not in gallery, skipping")
                continue
        self.samples = new_samples
        self.classes = list(self.label_map.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, camid = self.samples[idx]
        full_path = os.path.join(self.root_dir, img_path)

        with Image.open(full_path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)

        return img, label, camid


def load_model(path, cfg_file):
    model = FastReID(path, cfg_file)
    model.eval()
    model.cuda()
    return model.half().to("cuda")

def load_dataset(cfg):

    path = cfg.dataset
    transform_qg = transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    

    gallery_dataset = ReIDListDataset(path,
                                      f"{path}/gallery.txt",
                                      transform=transform_qg)
    query_dataset   = ReIDListDataset(path,
                                      f"{path}/query.txt",
                                      transform=transform_qg,
                                      relabel=False)
    query_dataset.relabel(gallery_dataset.label_map)

    query_loader = DataLoader(query_dataset, batch_size=512)
    gallery_loader = DataLoader(gallery_dataset, batch_size=512)
    
    return query_loader, gallery_loader

def extract_features(model, loader, feat_dim):
    model.eval()
    n_samples = len(loader.dataset)

    # Preallocate
    feats = torch.zeros((n_samples, feat_dim))
    labels = torch.zeros(n_samples, dtype=torch.long)
    
    camids = None
    """
    Debug
    camids = torch.zeros(n_samples, dtype=torch.long)
    """

    ptr = 0
    with torch.no_grad():
        for imgs, lbls, cams in loader:
            imgs = imgs.cuda()
            batch_feats = model(imgs)
            batch_feats = torch.nn.functional.normalize(batch_feats, p=2, dim=1).cpu()

            b = batch_feats.size(0)

            feats[ptr:ptr+b] = batch_feats
            labels[ptr:ptr+b] = lbls

            # -- Debugging Only -- #
            #print("min", batch_feats.min())
            #print("max", batch_feats.max())
            #if ptr > 512:
            #    exit(1)
            # -- -- #
            ptr += b

    return feats, labels, camids

def compute_cmc_map_in_gpu(query_feats,
                           query_ids,
                           gallery_feats,
                           gallery_ids,
                           batch_size=32092):
    query_feats = query_feats.to("cuda")
    query_ids   = query_ids.to("cuda")
    gallery_ids = gallery_ids.to("cuda")

    len_gallery_feats = len(gallery_feats)

    # Preallocation
    dist       = torch.empty(len_gallery_feats, device="cuda", dtype=gallery_feats.dtype)
    sorted_idx = torch.empty(len_gallery_feats, device="cuda", dtype=torch.int64)
    cmc        = torch.zeros(len(gallery_ids) , device="cuda", dtype=torch.int64)
    all_AP     = torch.tensor(0.0, device="cuda")
    valid_queries = torch.tensor(0, device="cuda", dtype=torch.int64)


    for i in range(len(query_feats)):
        queryf = query_feats[i:i+1]


        # Cos distance
        for j in range(0, len_gallery_feats, batch_size):
            galleryf = gallery_feats[j:j+batch_size].to("cuda")
            num_moved = galleryf.shape[0]
            sim = queryf @ galleryf.T
            dist[j:j+num_moved] = (1 - sim).squeeze(0)

        # mAP and Ranks
        sorted_idx = torch.argsort(dist)
        sorted_ids = gallery_ids[sorted_idx]

        q_id   = query_ids[i].item()
        y_true = (sorted_ids == q_id)
        tp = torch.cumsum(y_true, dim=0)
        total_positives = tp[-1]    # total sum

        if 0 == total_positives:
            print(f"Query {q_id}@{i+1} has no correct matches.")
            continue

        # Ranks
        rank = torch.where(y_true)[0][0]
        cmc[rank:] += 1

        # AP
        precision = tp / (torch.arange(len(y_true), device=y_true.device, dtype=torch.float32) + 1)
        ap = (precision * y_true).sum() / total_positives

        all_AP += ap
        valid_queries += 1


    if 0 == valid_queries.item():
        raise ValueError("Invalid queries, it's highly likely that the gallery doesn't contain any query's id.")

    all_AP = all_AP / valid_queries
    cmc = cmc / valid_queries

    cmc = cmc.cpu().numpy()
    mAP = all_AP.cpu()

    return cmc, mAP

"""
# Debugger: Install this to check if my implementation is correct:
#   pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
# from torchreid.metrics.rank import evaluate_rank
"""
def compute_metrics(cfg):
    
    model = load_model(cfg.model, cfg.model_cfg)
    query, gallery = load_dataset(cfg)

    feats_dim = 2048 # This is by design

    Q_feats, Q_ids, Q_cams = extract_features(model, query, feats_dim)
    G_feats, G_ids, G_cams = extract_features(model, gallery, feats_dim)

    """ Debug only
    G_cams = G_cams + 1
    distmat = 1 - torch.mm(Q_feats, G_feats.T).cpu().numpy()
    cmc_ref, map_ref = evaluate_rank(distmat, Q_ids, G_ids, Q_cams, G_cams, max_rank=50)
    print(f"ref mAP:{map_ref}, ref Rank: {cmc_ref[0]}")
    """

    return compute_cmc_map_in_gpu(
            Q_feats, Q_ids,
            G_feats, G_ids,
            batch_size=512000)

def usr_input():
    parser = argparse.ArgumentParser("Test mAP")

    parser.add_argument("--model",
                        required=True)

    parser.add_argument("--model_cfg",
                        required=True)

    parser.add_argument("--dataset",
                        required=True)

    return parser.parse_args()

if "__main__" == __name__:
    cfg = usr_input()
    cmc, mAP = compute_metrics(cfg)
    
    print(f"mAP: {mAP}, Rank-1: {cmc[0]}, Rank-5:{cmc[4]}, Rank-9:{cmc[9]}")

