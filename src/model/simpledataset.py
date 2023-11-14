import torch
import numpy as np


class SimpleDataset(torch.utils.data.Dataset):
    """
    A dataset that generates random embedded sequence and labels
    for decoder layers (no embedding layer or final classifier layer).
    """

    def __init__(self, seq, d_model, size=100):
        """_summary_

        Args:
            seq (int): Length of the sequence (context).
            d_model (int): Dimension of embedding.
            size (int, optional): Size of the dataset. Defaults to 100.
        """
        self._size = size
        self._inputs = np.random.randn(size, seq, d_model)
        self._labels = np.random.randn(size, seq)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return (
            torch.tensor(self._inputs[idx], dtype=torch.float32),
            self._labels[idx].astype("float32"),
        )


def main():
    dataset = SimpleDataset(1024, 768)
    item = dataset[0]
    x = item[0]
    y = item[1]
    print(x.shape, y.shape)


if __name__ == "__main__":
    main()
