from pathlib import Path

import pytest
import torch

from detr.data.coco_datamodule import COCODataModule
from detr.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [4, 8, 16, 32, 64, 128, 256])
def test_coco_datamodule(batch_size: int) -> None:
    """Tests `COCODataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    coco = COCODataModule(data_dir=data_dir, batch_size=batch_size)
    coco.prepare_data()

    assert not coco.data_train and not coco.data_val and not coco.data_test
    assert Path(data_dir, "train2017").exists()
    assert Path(data_dir, "val2017").exists()
    assert Path(data_dir, "annotations_trainval2017").exists()
    assert Path(
        data_dir, "annotations_trainval2017", "instances_train2017.json"
    ).exists()

    coco.setup()
    assert coco.data_train and coco.data_val and coco.data_test
    assert coco.train_dataloader() and coco.val_dataloader() and coco.test_dataloader()

    assert len(coco.data_train) + len(coco.data_test) == 118287
    assert len(coco.data_val) == 5000
    print(len(coco.data_train), len(coco.data_val), len(coco.data_test))

    batch = next(iter(coco.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert isinstance(y, list)
