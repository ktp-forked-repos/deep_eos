"""Provide implementation for data fetchers (like universal dependencies parsing."""

from enum import Enum
from typing import List

from deep_eos.data import Context
from deep_eos.utils import get_char_context
from deep_eos.utils import parse_dataset_to_buffer, parse_file_to_buffer


class Task(Enum):
    """Define parsing task."""

    UD_UNTOK = 'ud_untokenized'
    TEXT = 'text'


class EOSContextFetcher:  # pylint: disable=too-few-public-methods
    """Define methods for fetching contexts from datasets."""

    @staticmethod
    def fetch_contexts(task: Task,
                       base_path=None,
                       left_ws: int = 3,
                       right_ws: int = 3) -> List[Context]:
        """Fetch contexts from dataset.

        :param task: parsing task (like universal dependencies)
        :param base_path: path to dataset
        :param left_ws: left window size
        :param right_ws: right window size
        :return: contexts
        """
        contexts: List[Context]
        if task == Task.UD_UNTOK:
            buffer = parse_dataset_to_buffer(base_path)

        elif task == Task.TEXT:
            buffer = parse_file_to_buffer(base_path)

        contexts = list(get_char_context(left_window=left_ws,
                                         right_window=right_ws,
                                         buffer=buffer))
        return contexts
