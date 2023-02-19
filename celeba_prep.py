import os

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset


class CelebADataset(Dataset):

    # Custom dataset class for CelebA data

    def __init__(
        self,
        root_dir="./data",
        metadata_csv_name="list_attr_celeba.csv",
        split_csv_name="list_eval_partition.csv",
    ):

        self.root_dir = os.path.join(root_dir, "celebA")

        # Read in attributes
        self.attrs_df = pd.read_csv(os.path.join(self.root_dir, metadata_csv_name))

        # Split out filenames and attribute names
        self.data_dir = os.path.join(self.root_dir, "img_align_celeba")
        self.filename_array = self.attrs_df["image_id"].values
        self.attrs_df = self.attrs_df.drop(labels="image_id", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        # Get the y values (not blond = 0, blond = 1)
        target_idx = self.attr_idx("Blond_Hair")
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        # Get the confounder values (female = 0, male = 1)
        target_idx = self.attr_idx("Male")
        self.confounder_array = self.attrs_df[:, target_idx]

        # Groups:   0 --> not blond, female
        #           1 --> not blond, male
        #           2 --> blond, female
        #           3 --> blond, male
        self.n_groups = 4
        self.group_array = self.y_array * 2 + self.confounder_array

        # Read in train/val/test splits
        self.split_df = pd.read_csv(os.path.join(self.root_dir, split_csv_name))

        self.split_array = self.split_df["partition"].values
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        # Set transforms (no data augmentation)
        self.train_transform = get_transform_celebA()
        self.eval_transform = get_transform_celebA()

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        label = self.y_array[idx]
        group = self.group_array[idx]

        img_filename = os.path.join(self.data_dir, self.filename_array[idx])
        img = Image.open(img_filename).convert("RGB")

        if self.split_array[idx] == self.split_dict["train"]:
            img = self.train_transform(img)
        elif self.split_array[idx] in [self.split_dict["val"], self.split_dict["test"]]:
            img = self.eval_transform(img)

        return img, label, group, idx

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def split(self):

        # Splits the dataset into train, val and test data acc. to split_array

        train_indices = [
            i for i, x in enumerate(self.split_array) if x == self.split_dict["train"]
        ]
        val_indices = [
            i for i, x in enumerate(self.split_array) if x == self.split_dict["val"]
        ]
        test_indices = [
            i for i, x in enumerate(self.split_array) if x == self.split_dict["test"]
        ]

        return [
            Subset(self, indices)
            for indices in [train_indices, val_indices, test_indices]
        ]


def get_transform_celebA(augment=False):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)

    if not augment:
        transform = transforms.Compose(
            [
                transforms.CenterCrop(orig_min_dim),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=(0.7, 1.0),
                    ratio=(1.0, 1.3333333333333333),
                    interpolation=2,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transform
