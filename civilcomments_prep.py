import os

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from transformers import BertTokenizer


class CivilCommentsDataset(Dataset):

    # Costum dataset for CivilComments

    def __init__(
        self,
        data_dir="./data/CivilComments",
        data_filename="all_data_with_identities.csv",
        batch_size=16,
    ):

        self.data_dir = data_dir
        self.target_name = "toxicity"
        self.confounder_names = "identity_any"

        if batch_size == 32:
            self.max_length = 128
        elif batch_size == 24:
            self.max_length = 220
        elif batch_size == 16:
            self.max_length = 300
        else:
            assert False, "Invalid batch size"

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset."
            )

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, data_filename), index_col=0
        )

        # Labels:   y == 0 --> not toxic
        #           y == 1 --> toxic
        self.y_array = (self.metadata_df["toxicity"].values >= 0.5).astype("long")

        # Groups:   0 --> not toxic, no identities mentioned
        #           1 --> not toxic, identities mentioned
        #           2 --> toxic, no identities mentioned
        #           3 --> toxic, identities mentioned
        self.confounder_array = (
            (self.metadata_df.loc[:, "identity_any"] >= 0.5).values
        ).astype("int")
        self.group_array = (self.y_array * 2 + self.confounder_array).astype("int")

        # Extract splits
        self.split_dict = {"train": 0, "val": 1, "test": 2}
        for split in self.split_dict:
            self.metadata_df.loc[
                self.metadata_df["split"] == split, "split"
            ] = self.split_dict[split]

        self.split_array = self.metadata_df["split"].values

        # Extract text
        self.text_array = list(self.metadata_df["comment_text"])
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.y_array)

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        text = self.text_array[idx]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )  # 220
        x = torch.stack(
            (tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"]),
            dim=2,
        )
        x = torch.squeeze(x, dim=0)  # First shape dim is always 1

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
