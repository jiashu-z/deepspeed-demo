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
        self._inputs = torch.tensor(np.random.randn(size, seq, d_model), dtype=torch.float32)
        self._labels = torch.tensor(np.random.randn(size, seq), dtype=torch.float32)

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return (self._inputs[idx], self._labels[idx])
        # return (
            # torch.tensor(self._inputs[idx], dtype=torch.float32),
            # self._labels[idx].astype("float32"),
        # )


def main():
    dataset = SimpleDataset(1024, 768)
    x, y = dataset[0]
    print(x.shape, y.shape)
    print(x.size(), y.size)
    print(x.device)
    a, b = x.size()
    print(a, b)
    x = x.reshape(-1, *x.size())
    print(x.size())


if __name__ == "__main__":
    main()
