import argparse
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

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

    query_loader = DataLoader(query_dataset, batch_size=cfg.batch_sz)
    gallery_loader = DataLoader(gallery_dataset, batch_size=cfg.batch_sz)
    
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

            ptr += b

    return feats, labels, camids

def __cosine_batches():
    pass

def __compute_cmc_map_in_gpu(query_feats,
                           query_ids,
                           gallery_feats,
                           gallery_ids,
                           batch_size=0):

    len_gallery_feats = len(gallery_feats)
    len_query_feats   = len(query_feats)
    step_print = len_query_feats // 100

    query_feats = query_feats.to("cuda")
    query_ids   = query_ids.to("cuda")
    gallery_ids = gallery_ids.to("cuda")

    gallery_in_cpu = 1
    if batch_size == 0:
        gallery_feats = gallery_feats.to("cuda")
        batch_size = len_gallery_feats
        gallery_in_cpu = 1

    # Preallocation
    dist       = torch.empty(len_gallery_feats, device="cuda", dtype=gallery_feats.dtype)
    sorted_idx = torch.empty(len_gallery_feats, device="cuda", dtype=torch.int64)
    cmc        = torch.zeros(len(gallery_ids) , device="cuda", dtype=torch.int64)
    all_AP     = torch.tensor(0.0, device="cuda")
    valid_queries = torch.tensor(0, device="cuda", dtype=torch.int64)


    for i in range(len(query_feats)):
        queryf = query_feats[i:i+1]

        if i % step_print == 0:
            print(f"{i}/{len_query_feats}")

        # Cos distance
        for j in range(0, len_gallery_feats, batch_size):
            #galleryf = gallery_feats[j:j+batch_size].to("cuda")
            galleryf = gallery_feats[j:j+batch_size]
            if gallery_in_cpu: 
                galleryf = galleryf.to("cuda")
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

def __extract_features(cfg):
    model = load_model(cfg.model, cfg.model_cfg)
    query, gallery = load_dataset(cfg)

    feats_dim = 2048 # This is by design

    Q_feats, Q_ids, _ = extract_features(model, query, feats_dim)
    G_feats, G_ids, _ = extract_features(model, gallery, feats_dim)

    return Q_feats, Q_ids, G_feats, G_ids


def full_computation(cfg):

    Q_feats, Q_ids, G_feats, G_ids = __extract_features(cfg)
    
    return __compute_cmc_map_in_gpu(
            Q_feats, Q_ids,
            G_feats, G_ids,
            batch_size=cfg.batch_sz_mAP)

def __load_features(cfg):
    data = np.load(cfg.features)
    return data['q_feats'], data['q_ids'], data['g_feats'], data['g_ids']

def mAP_from_feats(cfg):

    Q_feats, Q_ids, G_feats, G_ids = __load_features(cfg)

    Q_feats = torch.from_numpy(Q_feats)
    G_feats = torch.from_numpy(G_feats)

    Q_ids = torch.from_numpy(Q_ids)
    G_ids = torch.from_numpy(G_ids)

    return __compute_cmc_map_in_gpu(
            Q_feats, Q_ids,
            G_feats, G_ids,
            batch_size=cfg.batch_sz_mAP)

def extract_save_features(cfg):

    Q_feats, Q_ids, G_feats, G_ids = __extract_features(cfg)

    Q_feats = Q_feats.numpy()
    Q_ids = Q_ids.numpy()
    G_feats = G_feats.numpy()
    G_ids = G_ids.numpy()

    np.savez_compressed(f"{cfg.out_dir}/{cfg.name}-features.npz",
        q_feats = Q_feats,
        q_ids = Q_ids,
        g_feats = G_feats,
        g_ids = G_ids
    )

    print("Q_feats size", Q_feats.shape)
    print("G_feats size", G_feats.shape)
    print("Q_ids size", Q_ids.shape)
    print("G_ids size", G_ids.shape)


def get_config():
    parser = argparse.ArgumentParser("Test mAP")

    parser.add_argument("--task",
                        type=str,
                        default="mAP-from-dataset")

    parser.add_argument("--out_dir",
                        type=str,
                        default=None)

    parser.add_argument("--name",
                        type=str,
                        default=None)

    parser.add_argument("--features",
                        type=str,
                        default=None)

    parser.add_argument("--model",
                        type=str,
                        default=None)

    parser.add_argument("--model_cfg",
                        type=str,
                        default=None)

    parser.add_argument("--dataset",
                        type=str,
                        default=None)

    parser.add_argument("--batch_sz",
                        type=int,
                        default=512)

    parser.add_argument("--batch_sz_mAP",
                        type=int,
                        default=0)

    cfg = parser.parse_args()

    match cfg.task:

        case "mAP-from-dataset":
            if cfg.model is None:
                raise Exception("A model is required, i.e. --model <path/to/model.pth>")

            if cfg.model_cfg is None:
                raise Exception("A model config file is required, i.e. --model_cfg <path/to/config.yaml>")

            if cfg.dataset is None:
                raise Exception("A dataset path is required, i.e. --dataset <path/to/dataset/dir>")

        case "features-only":
            if cfg.model is None:
                raise Exception("A model is required, i.e. --model <path/to/model.pth>")

            if cfg.model_cfg is None:
                raise Exception("A model config file is required, i.e. --model_cfg <path/to/config.yaml>")

            if cfg.dataset is None:
                raise Exception("A dataset path is required, i.e. --dataset <path/to/dataset/dir>")

            if cfg.out_dir is None:
                raise Exception("An Output directory is required, i.e. --out_dir <output/dir>")

            if cfg.name is None:
                raise Exception("An Output directory is required, i.e. --name <experiment name>")

        case "mAP-from-features":
            if cfg.features is None:
                raise Exception("Query Features numpy file is required, i.e. --features <path/to/features.npz>")

        case _:
            raise Exception("Uknown task {cfg.task}")

    return cfg

if "__main__" == __name__:
    cfg = get_config()

    match cfg.task:
        case "mAP-from-dataset":
            cmc, mAP = full_computation(cfg)

        case "features-only":
            extract_save_features(cfg)
            exit(0)

        case "mAP-from-features":
            cmc, mAP = mAP_from_feats(cfg)

        case _:
            raise Exception("Unknown task {cfg.task}")
    
    print(f"\nmAP: {mAP}, Rank-1: {cmc[0]}, Rank-5:{cmc[4]}, Rank-9:{cmc[9]}")

