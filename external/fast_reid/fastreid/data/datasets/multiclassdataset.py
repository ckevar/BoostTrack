from fast_reid.fastreid.data.datasets import DATASET_REGISTRY
from fast_reid.fastreid.data.datasets.bases import ImageDataset
import os
import numpy as np

def build_pid_map(root_dir, txt_files):

    data_name = os.path.dirname(txt_files[0]).split("/")[-1]
    subset_name = os.path.basename(txt_files[0])

    npz_file = f"{data_name}-{subset_name}-pid_map.npz"

    if os.path.isfile(npz_file):
        data = np.load(npz_file)
        pids = data["pids"]
        labels = data["labels"]
        # It's important to cast pid into int rather than keeping them into np type
        return {int(pid): int(label) for pid, label in zip(pids, labels)}

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

    np.savez(npz_file, pids=pids, labels=labels)

    return pid2label

def process_txt(root, txt_file, pid2label, subset="train"):
    """
    The Evaluation class is created everytime, so, this query and gallery text
    files are parsed at every evaluation and it takes some time do to (2.61 secs),
    which it's not that bad, but if it's called 600 times, then evaluation turns 
    into something larger. However, the gallery and query aren't that big so, they
    can actually fit in RAM. So, we attach a dictionary to the function. is this 
    safe? I dont know. But it makes it way faster.
    """

    if hasattr(process_txt, "datasets"):
        if subset in process_txt.datasets:
            return process_txt.datasets[subset]
    else:
        process_txt.datasets = {}

    data = []
    with open(txt_file) as f:
        for line in f:
            path, pid, camid = line.strip().split()
            pid = int(pid)
            camid = int(camid)
            data.append((os.path.join(root, path), 
                         pid2label[pid], 
                         camid))

    if subset != "train":
        # We dont need to hold the train values because it exist once
        process_txt.datasets[subset] = data
    
    return data


def sanitize_data(data):
    sanitized = []
    for i, (img_path, pid, camid) in enumerate(data):
        if pid is None:
            print(f"Skipping invalid PID at {img_path}")
            continue
        sanitized.append((img_path, pid, camid))
    return sanitized

class DatasetPaths(object):
    PATH = "/home/chris/Documents/Datasets/reid/fastreid/kittire_set"
    TRAIN_MAP = "train.txt"
    QUERY_MAP = "query.txt"
    GALLERY_MAP = "gallery.txt"
 
@DATASET_REGISTRY.register()
class MulticlassMOT17Train(ImageDataset):
    def __init__(self, root="datasets", **kwargs):
        dataset = DatasetPaths()

        self.root = os.path.join(root, dataset.PATH)
        txt = os.path.join(self.root, dataset.TRAIN_MAP)
        pid2label = build_pid_map(root, [txt])

        train = process_txt(self.root, txt, pid2label)

        assert len(train) > 0

        super().__init__(train, None, None, **kwargs)

@DATASET_REGISTRY.register()
class MulticlassMOT17Eval(ImageDataset):
   def __init__(self, root='datasets', **kwargs):
        dataset = DatasetPaths()

        self.root = os.path.join(root, dataset.PATH)
        query_txt   = os.path.join(self.root, dataset.QUERY_MAP)
        gallery_txt = os.path.join(self.root, dataset.GALLERY_MAP)

        pid2label = build_pid_map(root, [query_txt, gallery_txt])

        query   = process_txt(self.root, query_txt, pid2label, subset="query")
        gallery = process_txt(self.root, gallery_txt, pid2label, subset="gallery")

        assert len(query) > 0
        assert len(gallery) > 0

        super().__init__(None, query, gallery, **kwargs)

@DATASET_REGISTRY.register()
class MulticlassMOT17(ImageDataset):
    def __init__(self, root='datasets', **kwargs):
        dataset = DatasetPaths()

        self.root = os.path.join(root, dataset.PATH)
        # TODO: Edit this to point where the train.txt file is
        train_txt   = os.path.join(self.root, dataset.TRAIN_MAP)
        query_txt   = os.path.join(self.root, dataset.QUERY_MAP)
        gallery_txt = os.path.join(self.root, dataset.GALLERY_MAP)

        pid2label = build_pid_map(root, [train_txt, query_txt, gallery_txt])

        train   = process_txt(self.root, train_txt, pid2label)
        query   = process_txt(self.root, query_txt, pid2label)
        gallery = process_txt(self.root, gallery_txt, pid2label)

        gallery_pids = set(pid for _, pid, _ in gallery)
        filtered_query = []
        for item in query:
            if item[1] in gallery_pids:
                filtered_query.append(item)
            else:
                print(f"Query PID {item[1]} has no gallery match, skipping")
                exit(1)
        query = filtered_query

        assert len(train) > 0
        assert len(query) > 0
        assert len(gallery) > 0

        super().__init__(train, query, gallery, **kwargs)
