"""
Chalk-inspired Terminal Styling and Logging Engine for Synor AI.
Zero external dependencies, pure ANSI 256-color & style rendering.
"""

import sys
import time
from typing import Any, Dict, Optional


class _ChalkStyle:
    """Chainable ANSI text styler mirroring Chalk in JavaScript."""

    ANSI_CODES = {
        # Modifiers
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "italic": "\033[3m",
        "underline": "\033[4m",
        # Foreground Standard
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "gray": "\033[90m",
        # Foreground Bright
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        # Background Standard
        "bg_black": "\033[40m",
        "bg_red": "\033[41m",
        "bg_green": "\033[42m",
        "bg_yellow": "\033[43m",
        "bg_blue": "\033[44m",
        "bg_magenta": "\033[45m",
        "bg_cyan": "\033[46m",
        "bg_white": "\033[47m",
    }

    def __init__(self, codes=None):
        self._codes = codes or []

    def __getattr__(self, name: str) -> "_ChalkStyle":
        if name in self.ANSI_CODES:
            return _ChalkStyle(self._codes + [self.ANSI_CODES[name]])
        raise AttributeError(f"Invalid Chalk style '{name}'")

    def __call__(self, text: Any) -> str:
        s = str(text)
        if not self._codes:
            return s
        return "".join(self._codes) + s + self.ANSI_CODES["reset"]


# Singleton Chalk instance for fluent styling
chalk = _ChalkStyle()


# High-level logging & UI functions
def print_banner(title: str, subtitle: str = "", details: Optional[Dict[str, Any]] = None) -> None:
    """Print a modern, colorful CLI banner."""
    width = 62
    border = chalk.cyan("─" * width)
    print(f"\n{border}")
    print(f" {chalk.bold.bright_cyan('🧠 ' + title)}")
    if subtitle:
        print(f" {chalk.dim(subtitle)}")
    if details:
        print(f"{chalk.dim('─' * width)}")
        for k, v in details.items():
            print(f"  {chalk.bold.white(k)}: {chalk.bright_yellow(str(v))}")
    print(f"{border}\n")


def log_info(msg: str) -> None:
    timestamp = chalk.dim(time.strftime("%H:%M:%S"))
    badge = chalk.bold.bg_blue.white(" INFO ")
    print(f"{timestamp} {badge} {chalk.white(msg)}")


def log_success(msg: str) -> None:
    timestamp = chalk.dim(time.strftime("%H:%M:%S"))
    badge = chalk.bold.bg_green.black(" PASS ")
    print(f"{timestamp} {badge} {chalk.bright_green(msg)}")


def log_warn(msg: str) -> None:
    timestamp = chalk.dim(time.strftime("%H:%M:%S"))
    badge = chalk.bold.bg_yellow.black(" WARN ")
    print(f"{timestamp} {badge} {chalk.bright_yellow(msg)}")


def log_error(msg: str) -> None:
    timestamp = chalk.dim(time.strftime("%H:%M:%S"))
    badge = chalk.bold.bg_red.white(" ERR  ")
    print(f"{timestamp} {badge} {chalk.bright_red(msg)}")


def log_step(
    step: int,
    max_steps: int,
    train_loss: float,
    val_loss: float,
    lr: float,
    elapsed: float,
    is_best: bool = False,
    data_pct: Optional[float] = None,
) -> None:
    """Format and print an elegant training step record."""
    pct = (step / max_steps) * 100
    progress = f"{step:5d}/{max_steps:5d} ({pct:3.0f}%)"

    train_str = chalk.bright_white(f"{train_loss:.4f}")
    val_color = chalk.bold.bright_green if is_best else chalk.bright_white
    val_str = val_color(f"{val_loss:.4f}")
    lr_str = chalk.dim(f"{lr:.2e}")
    time_str = chalk.dim(f"{elapsed:4.1f}s")

    star = f" {chalk.bold.bright_yellow('★ New Record')}" if is_best else ""

    data_str = ""
    if data_pct is not None:
        data_str = f" │ {chalk.dim('File:')} {chalk.bold.bright_magenta(f'{data_pct:5.1f}%')}"

    print(
        f"  {chalk.bold.cyan('Step')} {chalk.yellow(progress)}{data_str} │ "
        f"{chalk.dim('Train:')} {train_str} │ "
        f"{chalk.dim('Val:')} {val_str} │ "
        f"{chalk.dim('LR:')} {lr_str} │ "
        f"{time_str}{star}"
    )


def print_user_prompt(prompt: str) -> None:
    user_tag = chalk.bold.bg_magenta.white(" USER ")
    print(f"\n{user_tag} {chalk.bright_white(prompt)}")


def print_ai_header() -> None:
    ai_tag = chalk.bold.bg_cyan.black(" SYNOR ")
    print(f"{ai_tag} ", end="", flush=True)
