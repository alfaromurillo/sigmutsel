# Contributing to sigmutsel

Thank you for considering contributing to sigmutsel!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/alfaromurillo/sigmutsel.git
cd sigmutsel
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

This project uses:
- **black** for code formatting (70 character line length)
- **ruff** for linting

Format your code before committing:
```bash
black src/ tests/
ruff check src/ tests/ --fix
```

## Pull Request Process

1. Create a new branch for your feature/fix
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Format your code
6. Submit a pull request

## Reporting Issues

Please use the GitHub issue tracker to report bugs or suggest
features.
