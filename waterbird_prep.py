import os

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, Subset


class WBDataset(Dataset):

    # Custom dataset class for waterbird data

    def __init__(
        self,
        data_dir="./data/waterbird_complete95_forest2water2",
        metadata_csv_name="metadata.csv",
    ):

        self.data_dir = data_dir

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Generate the dataset first."
            )

        # Read metadata
        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, metadata_csv_name))

        # Labels:   y == 0 --> Landbird
        #           y == 1 --> Waterbird
        self.y_array = self.metadata_df["y"].values

        # Groups:   0 --> LB on land
        #           1 --> LB on water
        #           2 --> WB on land
        #           3 --> WB on water
        self.confounder_array = self.metadata_df["place"].values
        self.group_array = (self.y_array * 2 + self.confounder_array).astype("int")

        # Extract filenames and splits
        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        # Set transforms (no data augmentation)
        self.train_transform = get_transform_wb()
        self.eval_transform = get_transform_wb()

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


def get_transform_wb(augment=False):
    scale = 256.0 / 224.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if not augment:
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (
                        int(224 * scale),
                        int(224 * scale),
                    )
                ),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (224, 224),
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    return transform
