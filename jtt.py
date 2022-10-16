import torch
from waterbird_prep import WBDataset 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64

# Loading full waterbird dataset
full_dataset = WBDataset(data_dir="./data/waterbird_complete95_forest2water2", 
    metadata_csv_name="metadata.csv")

# Splitting the full dataset into train, val, and test data according to the split column
# in metadata.csv
train_data, val_data, test_data = full_dataset.split()

#
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


# (Test) Display image and label.
train_features, train_labels = next(iter(train_dataloader))[:2]
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
