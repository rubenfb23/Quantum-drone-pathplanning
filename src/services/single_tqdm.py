"""Utility module for the SingleTqdm progress bar class."""

from tqdm import tqdm


class SingleTqdm(tqdm):
    """A single-position, non-clearing progress bar for threaded updates."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("position", 0)
        kwargs.setdefault("leave", True)
        super().__init__(*args, **kwargs)
