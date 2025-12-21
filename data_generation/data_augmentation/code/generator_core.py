"""Generic query generation helpers with plugin-based overrides."""

import importlib
import os
import sys
from typing import List, Optional

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_generation.shared.constants import Task, TaskType, get_generation_prompt  # type: ignore

PLUGIN_BASE = "data_generation.data_augmentation.code.task_plugins"


def _load_generation_plugin(task_type: TaskType):
    module_path = f"{PLUGIN_BASE}.{task_type.name}.prompts"
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError:
        return None


def build_generation_prompt(
    task: Task,
    text: str,
    examples: Optional[List[dict]] = None,
    narrative_focus: Optional[str] = None,
) -> str:
    """Build a generation prompt by first trying task-specific plugins.

    The fallback is the generic ``get_generation_prompt`` in ``shared.constants``.
    """

    plugin = _load_generation_plugin(task.task_type)
    for attr in ("build_generation_prompt", "get_generation_prompt"):
        if plugin and hasattr(plugin, attr):
            return getattr(plugin, attr)(
                task=task,
                text=text,
                examples=examples,
                narrative_focus=narrative_focus,
            )

    return get_generation_prompt(
        task=task,
        text=text,
        examples=examples,
        narrative_focus=narrative_focus,
    )
