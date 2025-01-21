import os
from typing import Any, Dict, Optional, Tuple
from zipfile import ZipFile

# from pycocotools.coco import COCO
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CocoDetection as COCO
from torchvision.transforms import transforms


class COCODataModule(LightningDataModule):
    """`LightningDataModule` for the COCO 2017 dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (100000, 5000, 18287),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `CocoDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

        # URLs for the COCO 2017 dataset
        self.urls = {
            "train": ("train2017", "http://images.cocodataset.org/zips/train2017.zip"),
            "val": ("val2017", "http://images.cocodataset.org/zips/val2017.zip"),
            "annotations": (
                "annotations_trainval2017",
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            ),
        }

        self.train_dir = os.path.join(data_dir, self.urls["train"][0])
        self.val_dir = os.path.join(data_dir, self.urls["val"][0])
        self.train_ann = os.path.join(
            data_dir, "annotations_trainval2017/instances_train2017.json"
        )
        self.val_ann = os.path.join(
            data_dir, "annotations_trainval2017/instances_val2017.json"
        )

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of COCO classes (80).
        """
        return 80

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # Download and extract the dataset
        for _, (folder, url) in self.urls.items():
            zip_path = os.path.join(self.hparams.data_dir, url.split("/")[-1])
            extract_path = os.path.join(self.hparams.data_dir, folder)

            if not os.path.exists(extract_path):
                # Download the zip file
                os.system(f"aria2c -x 16 -s 16 -k 1M -c {url}")

                # Extract the zip file
                with ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(self.hparams.data_dir)

                # Remove the zip file
                os.remove(zip_path)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = COCO(
                root=self.train_dir,
                annFile=self.train_ann,
                transform=self.transforms,
            )
            valset = COCO(
                root=self.val_dir,
                annFile=self.val_ann,
                transform=self.transforms,
            )
            # dataset = ConcatDataset(datasets=[trainset, testset])
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(42),
            # )
            self.data_train, self.data_test = random_split(
                dataset=trainset,
                lengths=[
                    self.hparams.train_val_test_split[0],
                    self.hparams.train_val_test_split[2],
                ],
                generator=torch.Generator().manual_seed(42),
            )
            self.data_val = valset

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.custom_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

    def custom_collate_fn(self, batch: Any) -> Any:
        """Custom collate function for the dataloader.

        :param batch: The batch to collate.
        :return: The collated batch.
        """
        images, targets = [], []
        for image, target in batch:
            images.append(image)
            targets.append(target)
        return torch.stack(images), targets


if __name__ == "__main__":
    data_dir = "data/"
    batch_size = 8

    coco = COCODataModule(data_dir=data_dir, batch_size=batch_size)
    coco.prepare_data()
    coco.setup()
    print("setup done")
