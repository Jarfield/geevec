"""Deprecated shim for backwards compatibility.

This module forwards to ``run_augmentation.py`` so existing entrypoints keep working.
"""

from run_augmentation import get_args, main  # noqa: F401


if __name__ == "__main__":
    main(get_args())
