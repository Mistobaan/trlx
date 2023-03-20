import json
import os
import time
from typing import Iterable

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from trlx.data.ppo_types import PPORLBatch, PPORLElement
from trlx.pipeline import BaseRolloutStore


class PPORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id):
        """
        Args:
            pad_token_id: The id of the token to use as padding.
        """
        super().__init__()

        self.pad_token_id = pad_token_id
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        """
        Add a list of expressions to the history.
        Args:
            exps: A list of expressions to add to the history.
        """
        self.history += exps

    def clear_history(self):
        """Clear the history of the current session."""
        self.history = []

    def export_history(self, location: str):
        """
        Export the history of the experiment to a json file.
        Args:
            location: The location to save the file.
        """
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            """
            Convert an experiment to a dictionary.
            Args:
                exp: An experiment.
            Returns:
                A dictionary.
            """
            {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> PPORLElement:
        """
        Args:
            index: The index of the element to return.
        Returns:
            The element at the given index.
        Raises:
            IndexError: If the index is out of range.
        """
        return self.history[index]

    def __len__(self) -> int:
        """Return the number of elements in the history."""
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """
        Args:
            batch_size: The batch size.
            shuffle: Whether to shuffle the data.
        Returns:
            A DataLoader for the dataset.
        """

        def collate_fn(elems: Iterable[PPORLElement]):
            """
            Args:
                elems: Iterable of PPORLElement
            Returns:
                PPORLBatch
            """
            return PPORLBatch(
                # Left padding of already left-padded queries
                pad_sequence(
                    [elem.query_tensor.flip(0) for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ).flip(1),
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)
