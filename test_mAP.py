import argparse
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

from external.adaptors.fastreid_adaptor import FastReID

# -- General Utils -- #

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

def extract_features(model, loader, feats_dim):
    model.eval()
    n_samples = len(loader.dataset)

    # Preallocate
    feats = torch.zeros((n_samples, feats_dim))
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

def __gen_transforms():
    return transforms.Compose([
        transforms.Resize((128, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# -- Utils for Distances -- #

def __get_basename(filename_plus_extension):
    fe = filename_plus_extension.split(".")
    ext = fe[1]
    basename = fe[0]
    if basename is None:
        print(f"Filename {basename} seems off. Saving with temporary name 'tmp'")
        basename = "tmp"

    return basename

def __load_dataset_for_dists(cfg):
    path = cfg.dataset
    map_file = cfg.map
    
    trans = __gen_transforms()
    
    dataset = ReIDListDataset(path,
                              f"{path}/{map_file}",
                              transform=trans,
                              relabel=False)

    loader = DataLoader(dataset, batch_size=cfg.batch_sz)

    return loader

def __extract_features_for_dists(cfg):
    model = load_model(cfg.model, cfg.model_cfg)
    dataset = __load_dataset_for_dists(cfg)

    feats, ids, _ = extract_features(model, dataset, cfg.feats_dim)

    return feats, ids

## -- Utils for __compute_distances  -- ##

def ___centered_features(feats, inverse_indices, num_groups):

    sum_feats = torch.zeros(num_groups, feats.size(1), device=feats.device, dtype=feats.dtype)
    sum_feats.index_add_(0, inverse_indices, feats)

    counts = torch.bincount(inverse_indices).float().unsqueeze(1)

    feats_mean = sum_feats / counts

    del sum_feats

    return feats_mean, counts



def ___row_distances(feats, feats_mean, inverse_indices):
    expanded_means = feats_mean[inverse_indices]

    if "cpu" in feats.device:
        feats = feats.to("cuda")
 
    dot_product = (feats * expanded_means).sum(1)
    row_distances = 1 - dot_product
    return row_distances


def ___row_distances_2gpu(feats_cpu, feats_mean, inverse_indices):
    if "cuda" in feats_cpu.device:
        feats_cpu = feats_cpu.to("cpu")
    #feats_mean2 = feat_mean.to("cuda")
    #inverse_indices2 = inverse_indices.to("cuda")

    expanded_means = feats_mean[inverse_indices] # expanded_means the same size as feats_cpu

    #del feats_mean2, inverse_indices2

    expanded_means = expanded_means.to("cpu")

    # Streaming Multiplication
    N = feats_cpu.size(0)
    device = "cuda:0"
    row_distances = torch.empty(N, feats.size(1), device=device, dtype=feats_cpu.dtype)

    batch_sz = 100000 # This can be part of configuration

    for i in range(0, N, batch_sz):

        # Define the slice
        end = min(i + batch_sz, N)

        # Move batches to GPU
        batch_1 = feats[i:end].to(device)
        batch_2 = expanded_means[i:end].to(devices)

        # COmpute on GPU and keep them
        batch_r = batch_1 * batch_2
        row_distances[i:end] = batch_r

        # Free GPU
        del batch_1,  batch_2, batch_r

    row_distances = row_distances.sum(1)
    row_distances = 1 - row_distances

    return row_distances

def ___unique_ids(ids):
    gpu_ids = ids.to("cuda")

    uniq_ids, inverse_indices = torch.unique(gpu_ids, return_inverse=True)
    num_groups = uniq_ids.size(0)
    
    del gpu_ids

    return uniq_ids, inverse_indices, num_groups

def __mean_feats_vectorized(feats, ids):

    uniq_ids, inverse_indices, num_groups = ___unique_ids(ids)

    feats = feats.to("cuda")
    # -- Centered Features -- #
    feats_mean, counts = ___centered_features(feats, inverse_indices, num_groups)

    if cfg.num_devices == 1:
    # -- Compute Distances -- #
        row_distances = ___row_distances(feats, feats_mean, inverse_indices)
    elif cfg.num_devices == 2:
        feats = feats.to("cpu")
        row_distances = ___row_distances_2gpu(feats, feats_mean, inverse_indices)
    else:
        del feats, uniq_ids, inverse_indices, num_groups, ids
        raise ValueError(f"Number of devices {cfg.num_devices} not supported. min: 1, max: 2.")

    # -- Average, min, max distances -- #
    sum_dist = torch.zeros(num_groups, device="cuda", dtype=feats_mean.dtype)
    max_dist = torch.empty(num_groups, device="cuda", dtype=feats_mean.dtype)
    min_dist = torch.empty(num_groups, device="cuda", dtype=feats_mean.dtype)

    max_dist.fill_(float('-inf'))
    min_dist.fill_(float('inf'))

    sum_dist.index_add_(0, inverse_indices, row_distances)
    max_dist.scatter_reduce_(0, inverse_indices, row_distances, reduce='amax', include_self=False)
    min_dist.scatter_reduce_(0, inverse_indices, row_distances, reduce='amin', include_self=False)

    sum_dist = sum_dist / counts.squeeze()

    # -- Sort IDs by average distances -- #
    sorted_idx = torch.argsort(sum_dist, descending=True)
    uniq_ids = uniq_ids[sorted_idx]
    feats_mean = feats_mean[sorted_idx]
    sum_dist = sum_dist[sorted_idx]
    min_dist = min_dist[sorted_idx]
    max_dist = max_dist[sorted_idx]

    return uniq_ids, feats_mean, sum_dist, min_dist, max_dist


def __inter_id_dists_vectorized(anchor_feats, anchor_ids):
    dist = 1 - anchor_feats @ anchor_feats.T # Tensor size patches x ids
    min_dist, min_indices = torch.min(dist, dim=1)

    row_indices = torch.arange(dist.size(0), device=dist.device)
    mask = min_indices != row_indices 

    confused_ids = anchor_ids[mask]
    distractor_ids = anchor_ids[min_indices][mask]

    confusing_dist = min_dist[mask]

    return confused_ids, distractor_ids, confusing_dist

def __save_intra(file, ids, dists, min_d, max_d):
    filename = f"{file}-intra-dist.txt"
    with open(filename, 'w') as fd:
        fd.write("# id mean_dist min_dist max_dist\n")
        for i, d, md, MD in zip(ids, dists, min_d, max_d):
            fd.write(f"{i} {d:.6f} {md:.6f} {MD:.6f}\n")

def __save_inter(file, c_ids, d_ids, dists):
    filename = f"{file}-inter-dist.txt"
    with open(filename, 'w') as fd:
        fd.write("# confused_id distractor_id confusing_dist\n")
        for ci, di, dd in zip(c_ids, d_ids, dists):
            fd.write(f"{ci} {di} {dd:.6f}\n")



def __compute_distances(cfg, feats, ids):
    
    print("Computing intra ID distances...")
    
    if cfg.num_devices == 1:
        u_ids, feats_mean, dists, min_d, max_d = __mean_feats_vectorized(feats, ids)
    elif cfg.num_devices == 2:
        u_ids, feats_mean, dists, min_d, max_d = __mean_feats_vectorized_2gpus(feats, ids)
    else:
        raise ValueError(f"Number of devices {cfg.num_devs} not supported. min: 1, max: 2.")
    print("Computing inter ID distances...")
    confused_ids, distractor_ids, confusing_d = __inter_id_dists_vectorized(feats_mean, u_ids)

    print(f"Saving in {cfg.out_file}")
    u_ids = u_ids.to("cpu").numpy()
    dists = dists.to("cpu").numpy()
    min_d = min_d.to("cpu").numpy()
    max_d = max_d.to("cpu").numpy()
    confused_ids = confused_ids.to("cpu").numpy()
    distractor_ids = distractor_ids.to("cpu").numpy()
    confusing_d = confusing_d.to("cpu").numpy()

    __save_intra(cfg.out_file, u_ids, dists, min_d, max_d)
    __save_inter(cfg.out_file, confused_ids, distractor_ids, confusing_d)

# -- Utils for mAP -- #
def __load_dataset_for_mAP(cfg):

    path = cfg.dataset
    trans_qg = __gen_transforms()

    gallery_dataset = ReIDListDataset(path,
                                      f"{path}/gallery.txt",
                                      transform=trans_qg)
    query_dataset   = ReIDListDataset(path,
                                      f"{path}/query.txt",
                                      transform=trans_qg,
                                      relabel=False)
    query_dataset.relabel(gallery_dataset.label_map)

    query_loader = DataLoader(query_dataset, batch_size=cfg.batch_sz)
    gallery_loader = DataLoader(gallery_dataset, batch_size=cfg.batch_sz)
    
    return query_loader, gallery_loader


def __compute_cmc_map_in_gpu(query_feats, query_ids,
                             gallery_feats, gallery_ids,
                             batch_size=0):
    """
    # Debugger: Install this to check if my implementation is correct:
    #   pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
    # from torchreid.metrics.rank import evaluate_rank
    """

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

        # Uncomment this for large datasets if i % step_print == 0: print(f"{i}/{len_query_feats}")

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

def __extract_features_for_mAP(cfg):
    model = load_model(cfg.model, cfg.model_cfg)
    query, gallery = __load_dataset_for_mAP(cfg)

    Q_feats, Q_ids, _ = extract_features(model, query, cfg.feats_dim)
    G_feats, G_ids, _ = extract_features(model, gallery, cfg.feats_dim)

    return Q_feats, Q_ids, G_feats, G_ids

# -- Utils for mAP_from_feats -- #
def __load_features_for_mAP(cfg):
    data = np.load(cfg.features)
    return data['q_feats'], data['q_ids'], data['g_feats'], data['g_ids']


# -- Interfaces -- #

def distances_from_dataset(cfg):
    print("\nExtracting features...")
    feats, ids = __extract_features_for_dists(cfg)
    __compute_distances(cfg, feats, ids)

def distances_from_features(cfg):
    data = np.load(cfg.features)

    ids = torch.from_numpy(data['ids'])
    feats = torch.from_numpy(data['feats'])

    __compute_distances(cfg, feats, ids)
    
def extract_save_features_for_dists(cfg):
    feats, ids = __extract_features_for_dists(cfg)
    feats = feats.numpy()
    ids = ids.numpy()
    np.savez_compressed(
        f"{cfg.out_file}-features.npz",
        feats = feats,
        ids = ids
    )

def mAP_from_dataset(cfg):

    Q_feats, Q_ids, G_feats, G_ids = __extract_features_for_mAP(cfg)
    
    cmc, mAP = __compute_cmc_map_in_gpu(
            Q_feats, Q_ids,
            G_feats, G_ids,
            batch_size=cfg.batch_sz_mAP)

    print(f"\nmAP: {mAP}| Rank-1: {cmc[0]}| Rank-5: {cmc[4]}| Rank-9: {cmc[8]}")

def mAP_from_feats(cfg):

    Q_feats, Q_ids, G_feats, G_ids = __load_features_for_mAP(cfg)

    Q_feats = torch.from_numpy(Q_feats)
    G_feats = torch.from_numpy(G_feats)

    Q_ids = torch.from_numpy(Q_ids)
    G_ids = torch.from_numpy(G_ids)

    cmc, mAP =__compute_cmc_map_in_gpu(
            Q_feats, Q_ids,
            G_feats, G_ids,
            batch_size=cfg.batch_sz_mAP)

    print(f"\nmAP {mAP}| Rank-1 {cmc[0]}| Rank-5 {cmc[4]}| Rank-9 {cmc[9]}")

def extract_save_features_for_mAP(cfg):

    Q_feats, Q_ids, G_feats, G_ids = __extract_features_for_mAP(cfg)

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

    print("Fallowing Information is important to determine GPU batch upon calculating mAP in batches if needed.")
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
    parser.add_argument("--map",
                        type=str,
                        default="train.txt")
    
    parser.add_argument("--feats_dim",
                        type=int,
                        default=2048, # This is usually the case.
                        help="2048 is usually the case in fastreid models. But it might vary."
                        )

    parser.add_argument("--num_devices",
                        type=int,
                        default=1,
                        help="Number of devices.")

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

        # -- DISTANCES, useful to mine hard IDs -- #
        case "distances-features-only":
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

            print(f"\nWARNING: Computing distances on map file {cfg.map}, you can set upload different map. --map train.txt|gallery.txt|query.txt. Keep in mind that query set contains only one image per ID\n")

        case "distances-from-dataset":

            if cfg.model is None:
                raise Exception("A model is required, i.e. --model <path/to/model.pth>")

            if cfg.model_cfg is None:
                raise Exception("A model config file is required, i.e. --model_cfg <path/to/config.yaml>")

            if cfg.dataset is None:
                raise Exception("A dataset path is required, i.e. --dataset <path/to/dataset/dir>")

            if cfg.name is None:
                raise Exception("An Output directory is required, i.e. --name <experiment name>")

            print(f"\nWARNING: Computing distances on map file {cfg.map}, you can set upload different map. --map train.txt|gallery.txt|query.txt\n")

            if cfg.out_dir is None:
                raise Exception("An Output directory is required, i.e. --out_dir <output/dir>")

        case "distances-from-features":
            if cfg.features is None:
                raise Exception("Query Features numpy file is required, i.e. --features <path/to/features.npz>")
            if cfg.out_dir is None:
                raise Exception("An Output directory is required, i.e. --out_dir <output/dir>")

        case _:
            raise Exception("Uknown task {cfg.task}")
    
    if not os.path.isdir(cfg.out_dir):
        raise ValueError(f"Output dir {cfg.out_dir} doesn't exist.")
    
    # -- Distance Files -- #
    if "distances" in cfg.task:
        map_file = __get_basename(cfg.map)
        cfg.out_file = f"{cfg.out_dir}/{map_file}-{cfg.name}"
        
    return cfg

if "__main__" == __name__:
    cfg = get_config()

    match cfg.task:
        # -- mAP -- #
        case "mAP-from-dataset":
            mAP_from_dataset(cfg)

        case "features-only":
            extract_save_features_for_mAP(cfg)

        case "mAP-from-features":
            mAP_from_feats(cfg)

        # -- Hard ID mining -- #

        case "distances-from-dataset":
            distances_from_dataset(cfg)

        case "distances-features-only":
            extract_save_features_for_dists(cfg)

        case "distances-from-features":
            distances_from_features(cfg)

        case _:
            raise ValueError(f"Unknown task {cfg.task}")
            exit(1)
