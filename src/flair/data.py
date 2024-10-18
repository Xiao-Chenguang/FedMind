from typing import Sized

from torch.utils.data import Dataset


class ClientDataset(Dataset):
    """
    A Dataset class wrapper to store the indices of the data samples that belong to a client.
    """

    def __init__(self, org_dataset: Dataset, idxs: list):
        """Initialize the ClientDataset object.

        Args:
            org_dataset: The original dataset.
            idxs: The indices of the data samples that belong to the client.
        """
        self.org_dataset = org_dataset
        self.idxs = idxs

        # Check if the indices are valid
        assert len(self.idxs) > 0, "The client dataset is empty."
        assert isinstance(org_dataset, Sized), "org_dataset must be Sized."
        assert max(self.idxs) < len(org_dataset), "The maximum index is out of bounds."
        assert min(self.idxs) >= 0, "The minimum index is out of bounds."

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.org_dataset[self.idxs[item]]
