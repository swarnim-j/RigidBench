from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

from . import __version__

_COMMANDS = {
    "download": ("rigidbench.data", "Download and extract the evaluation set", None),
    "inputs": ("rigidbench.benchmark", "Export the exact model inputs", None),
    "validate": ("rigidbench.eval.validate", "Check a directory of generated videos", None),
    "evaluate": ("rigidbench.eval.run", "Evaluate a directory of generated videos", "eval"),
    "generate": (
        "rigidbench.eval.generate.cli",
        "Reproduce generations from supported models",
        "wan, cosmos, or replicate",
    ),
    "assets": ("rigidbench.scenes.tools.download", "Download the configured simulator assets", None),
    "render": ("rigidbench.scenes.run", "Render simulator data", "scenes"),
    "preprocess": ("rigidbench.train.data.preprocess", "Encode clips for fine-tuning", "train"),
    "train": ("rigidbench.train.run", "Fine-tune a video model", "train"),
    "probe": ("rigidbench.probe.__main__", "Run representation experiments", "train and probe"),
}

_GROUPS = (
    ("commands", ("download", "inputs", "validate", "evaluate")),
    ("reproduction", ("generate",)),
    ("research", ("assets", "render", "preprocess", "train", "probe")),
)


def _print_help() -> None:
    print("usage: rigidbench <command> [options]\n")
    width = max(len(name) for name in _COMMANDS)
    for heading, commands in _GROUPS:
        print(f"{heading}:")
        for name in commands:
            _, description, _ = _COMMANDS[name]
            print(f"  {name:<{width}}  {description}")
        print()
    print("Run 'rigidbench <command> --help' for command-specific options.")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if args[0] in {"-V", "--version"}:
        print(f"rigidbench {__version__}")
        return 0

    command = args.pop(0)
    if command not in _COMMANDS:
        print(f"rigidbench: unknown command '{command}'\n", file=sys.stderr)
        _print_help()
        return 2

    module_name, _, extra = _COMMANDS[command]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name and not error.name.startswith("rigidbench") and extra:
            print(
                f"The '{command}' command is missing the '{error.name}' dependency. "
                f"Install the {extra} extra described in the README.",
                file=sys.stderr,
            )
            return 1
        raise

    sys.argv = [f"rigidbench {command}", *args]
    result = module.main()
    return int(result) if isinstance(result, int) else 0
