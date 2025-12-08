"""ORYX CLI entry point for module execution.

This allows running ORYX as a module:
    python -m oryx --help
    python -m oryx run --seed "contractors" --audience "businesses"
"""

from .cli import app

if __name__ == "__main__":
    app()
