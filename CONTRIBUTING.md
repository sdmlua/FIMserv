# Contributing to FIMserv

Thank you for your interest in contributing to **FIMserv**! This tool is developed and maintained by the Surface Dynamics Modeling Lab (SDML) at The University of Alabama. Contributions of all kinds are welcome — bug reports, feature requests, documentation improvements, and code.

## Getting Started

1. **Fork** the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/FIMserv.git
   cd FIMserv
   ```

2. **Create a virtual environment** (recommended) and install the package in editable mode with the development dependencies:

   ```bash
   conda create --name fimserve python==3.10
   conda activate fimserve

   pip install -e ".[dev]"
   ```

3. **Create a branch** for your change:

   ```bash
   git checkout -b my-feature
   ```

## Code Style and Linting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. The configuration lives in `pyproject.toml`.

Before opening a pull request, run:

```bash
# Lint the codebase (auto-fix what is safe to fix)
ruff check src tests --fix

# Format the codebase
ruff format src tests
```

Both commands must pass cleanly for a pull request to be accepted.

## Running Tests

Tests are written with [pytest](https://docs.pytest.org/) and live in the `tests/` directory:

```bash
pytest
```

Please add or update tests for any code you change, and make sure the existing test suite still passes.

## Submitting a Pull Request

1. Keep pull requests focused — one feature or fix per PR.
2. Write a clear description of **what** the change does and **why** it is needed.
3. Make sure `ruff check`, `ruff format --check`, and `pytest` all pass.
4. Reference any related issues (e.g., `Fixes #123`).

## Reporting Issues

Found a bug or have a feature request? Please open an issue at
[github.com/sdmlua/FIMserv/issues](https://github.com/sdmlua/FIMserv/issues) and include:

- A clear description of the problem or request
- Steps to reproduce (for bugs), including the HUC8 ID and date range used
- Your Python version and operating system

## Contact

- Dr. Sagy Cohen (sagy.cohen@ua.edu)
- Dr. Anupal Baruah (abaruah@ua.edu)
- Supath Dhital (sdhital@crimson.ua.edu)
