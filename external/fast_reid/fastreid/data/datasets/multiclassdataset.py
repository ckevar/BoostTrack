from fast_reid.fastreid.data.datasets import DATASET_REGISTRY
from fast_reid.fastreid.data.datasets.bases import ImageDataset
import os
import numpy as np

def build_pid_map(root_dir, txt_files):

    if os.path.isfile("pid_map.npz"):
        data = np.load("pid_map.npz")
        pids = data["pids"]
        labels = data["labels"]
        return {pid: label for pid, label in zip(pids, labels)}

    all_pids = set()
    for txt in txt_files:
        with open(txt) as f:
            for line in f:
                _, pid, _ = line.strip().split()
                all_pids.add(int(pid))

    pid2label = {pid: idx for idx, pid in enumerate(sorted(all_pids))}

    # Store PIDs to avoid recomputation
    pids = np.array(list(pid2label.keys()), dtype=np.int64)
    labels = np.array(list(pid2label.values()), dtype=np.int64)

    np.savez("pid_map.npz", pids=pids, labels=labels)

    return pid2label

def process_txt(root, txt_file, pid2label):
    data = []
    with open(txt_file) as f:
        for line in f:
            path, pid, camid = line.strip().split()
            pid = int(pid)
            camid = int(camid)
            data.append((os.path.join(root, path), 
                         pid2label[pid], 
                         camid))

    return data

@DATASET_REGISTRY.register()
class MulticlassMOT17(ImageDataset):
    dataset_dir = "/home/chris/Documents/Datasets/reid/fastreid/mot17re_set"

    def __init__(self, root='datasets', **kwargs):
        self.root = os.path.join(root, self.dataset_dir)
        # TODO: Edit this to point where the train.txt file is
        train_txt   = os.path.join(self.root, "train.txt")
        query_txt   = os.path.join(self.root, "query.txt")
        gallery_txt = os.path.join(self.root, "gallery.txt")

        pid2label = build_pid_map(root, [train_txt, query_txt, gallery_txt])

        train   = process_txt(self.root, train_txt, pid2label)
        query   = process_txt(self.root, query_txt, pid2label)
        gallery = process_txt(self.root, gallery_txt, pid2label)

        super().__init__(train, query, gallery, **kwargs)
