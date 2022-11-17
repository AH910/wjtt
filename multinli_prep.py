import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


class MultiNLIDataset(Dataset):

    # Custom dataset class for MultiNLI data

    def __init__(
        self,
        data_dir="./data/multiNLI",
        metadata_csv_name="metadata_preset.csv",
    ):

        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset."
            )

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, metadata_csv_name), index_col=0
        )

        # Labels:   y == 0 --> contradiction
        #           y == 1 --> entailment
        #           y == 2 --> neutral
        self.y_array = self.metadata_df["gold_label"].values

        # Groups:   0 --> contradiction, 2nd sentence has no negation
        #           1 --> contradiction, 2nd sentence has negation
        #           2 --> entailment, 2nd sentence has no negation
        #           3 --> entailment, 2nd sentence has negation
        #           4 --> neutral, 2nd sentence has no negation
        #           5 --> neutral, 2nd sentence has negation
        self.confounder_array = self.metadata_df["sentence2_has_negation"].values
        self.group_array = (self.y_array * 2 + self.confounder_array).astype("int")

        # Extract splits
        self.split_array = self.metadata_df["split"].values
        self.split_dict = {"train": 0, "val": 1, "test": 2}

        # Load features
        self.features_array = []
        for feature_file in [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]:

            features = torch.load(os.path.join(self.data_dir, feature_file))

            self.features_array += features

        self.all_input_ids = torch.tensor(
            [f.input_ids for f in self.features_array], dtype=torch.long
        )
        self.all_input_masks = torch.tensor(
            [f.input_mask for f in self.features_array], dtype=torch.long
        )
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in self.features_array], dtype=torch.long
        )
        self.all_label_ids = torch.tensor(
            [f.label_id for f in self.features_array], dtype=torch.long
        )

        self.x_array = torch.stack(
            (self.all_input_ids, self.all_input_masks, self.all_segment_ids), dim=2
        )

        assert np.all(np.array(self.all_label_ids) == self.y_array)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        x = self.x_array[idx, ...]
        return x, y, g, idx

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
