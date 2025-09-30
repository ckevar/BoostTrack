from fast_reid.fastreid.data.datasets import DATASET_REGISTRY
from fast_reid.fastreid.data.datasets.bases import ImageDataset
import os

@DATASET_REGISTRY.register()
class MulticlassMOT17(ImageDataset):
    dataset_dir = "/home/chris/Documents/Datasets/reid/fastreid/mot17re_set"

    def __init__(self, root='datasets', **kwargs):
        self.root = os.path.join(root, self.dataset_dir)
        # TODO: Edit this to point where the train.txt file is
        train = self._load_list('train.txt')
        query = self._load_list('query.txt')
        gallery = self._load_list('gallery.txt')

        super().__init__(train, query, gallery, **kwargs)

    def _load_list(self, fname):
        data = []
        with open(os.path.join(self.root, fname)) as f:
            for line in f:
                img_path, pid, camid = line.strip().split()
                data.append((os.path.join(self.root, img_path), int(pid), int(camid)))
        return data

