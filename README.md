# Multihop QA Agent

A Python‑based multi‑hop question‑answering (QA) agent that combines retrieval‑augmented generation with chain‑of‑thought reasoning to answer complex queries over large document collections.

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Multi‑hop reasoning** – traverses multiple knowledge sources to synthesize answers.
- **Retrieval‑augmented generation** – integrates external documents via dense retrieval.
- **Modular design** – clean separation of configs, data, Docker, source code, and tests.
- **Docker support** – containerised environment for reproducibility.

---

## Installation

### Prerequisites
- Python 3.10 or newer
- `virtualenv` (recommended)
- Docker (optional, for containerised run)

### Local setup
```bash
# Clone the repo
git clone https://github.com/bhavsarnaitik4/Multi-Hop-QA-Agent.git
cd Multi-Hop-QA-Agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

If you prefer Docker:
```bash
docker build -t multihop-qa-agent ./docker
docker run -it multihop-qa-agent
```

---

## Quick Start
```bash
python -m src.main --query "What are the main challenges in multi‑hop QA?"
```
*(Adjust the command to match the actual entry point in `src/`.)*

---

## Project Structure
```
├── .gitignore
├── .env                # Example env file (not tracked)
├── configs/            # Configuration files
├── data/               # Sample data / corpora
├── docker/             # Dockerfile and scripts
├── src/                # Core source code
├── tests/              # Test suite
├── README.md           # ↖️ This file
└── requirements.txt    # Python dependencies
```

---

## Configuration
All configurable parameters live under `configs/`. Typical files:
- `configs/model.yaml` – select the LLM backend.
- `configs/retriever.yaml` – vector store / index settings.

Update `.env` with any required API keys (keep it out of version control).

---

## Testing
```bash
pytest tests
```

---

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b my-feature`).
3. Write tests for your changes.
4. Open a Pull Request.

---

## License
Distributed under the MIT License. See `LICENSE` for details.

---

## Contact
- Author: **Naitik Bhavsar** – https://github.com/bhavsarnaitik4
- Issues: Use the GitHub *Issues* tab.
