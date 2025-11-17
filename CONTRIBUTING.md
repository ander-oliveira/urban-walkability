# Contributing Guidelines

Thank you for your interest in contributing to the Urban Walkability Analysis project! This document provides guidelines for contributing to this research software.

---

## Code of Conduct

### Our Standards

- **Respectful communication**: Be kind and constructive

- **Collaborative spirit**: Help others learn and grow

- **Scientific integrity**: Ensure accuracy and reproducibility

- **Open mindset**: Welcome diverse perspectives and approaches

### Reporting Issues

Report isues to: ander.oliveira@ime.usp.br

---

## Getting Started

### Prerequisites

- Python 3.12.5

- Git version control

- Basic understanding of geospatial analysis

- Familiarity with OSMnx and GeoPandas

### Fork and Clone

```bash

# Fork the repository on GitHub

# Then clone your fork

git  clone  https://github.com/yourusername/urban-walkability.git

cd  urban-walkability



# Add upstream remote

git  remote  add  upstream  https://github.com/original/urban-walkability.git

```

### Development Setup

```bash

# Create virtual environment

python  -m  venv  venv

source  venv/bin/activate  # Windows: venv\Scripts\activate



# Install dependencies

pip  install  -r  requirements.txt


```

---

## Development Workflow

### 1. Create Feature Branch

```bash

# Update your fork

git  checkout  main

git  pull  upstream  main



# Create feature branch

git  checkout  -b  feature/your-feature-name

```

### 2. Make Changes

- Write code following PEP 8 guidelines

- Add docstrings to all functions

- Include type hints

### 3. Commit Changes

```bash

# Stage changes

git  add  <modified-files>



# Commit with descriptive message

git  commit  -m  "message:some message"

```

**Commit message format:**

- `feat:` New feature

- `fix:` Bug fix

- `docs:` Documentation changes

- `refactor:` Code restructuring

- `perf:` Performance improvement

- `test:` Test additions/changes

### 4. Push and Create PR

```bash

# Push to your fork

git  push  origin  feature/your-feature-name



# Open Pull Request on GitHub

# - Describe changes clearly

# - Reference related issues

# - Request review from maintainers

```

---

## After Merge

- Delete feature branch

- Update your local repository

---

## Resources

- [Python Official Docs](https://docs.python.org/3/)

- [OSMnx Documentation](https://osmnx.readthedocs.io/)

- [GeoPandas User Guide](https://geopandas.org/)

- [H3 Documentation](https://h3geo.org/docs/)

---

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0 (GPLv3).
