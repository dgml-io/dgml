# Contributing to DGML

DGML is an open initiative founded by Docugami and Inveniam. This repository
contains the Python reference implementation of the DGML format. The format
specification itself lives in [dgml-io/dgml-spec](https://github.com/dgml-io/dgml-spec).

## How to contribute

### Ask a question or start a discussion
Open a [GitHub Discussion](../../discussions) for questions about the
implementation, ideas for new features, or anything else. For questions about
the format itself, head to [dgml-io/dgml-spec](https://github.com/dgml-io/dgml-spec/discussions).

### Report a bug
Open a [GitHub Issue](../../issues) with a clear description of the problem,
steps to reproduce, and the version you are using. If the bug is in the spec
rather than the implementation, open the issue in
[dgml-io/dgml-spec](https://github.com/dgml-io/dgml-spec/issues) instead.

### Propose a change
1. Open an Issue first to describe what you want to change and why
2. Once there is consensus, submit a Pull Request
3. Small fixes (typos, bugs with obvious solutions) can go straight to a PR

### Development setup
```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy packages
```

Run `scripts/verify.sh` before submitting — it runs the same checks as CI.

## Contributor License Agreement (CLA)

Before your Pull Request can be merged, you must sign the Apache Individual
Contributor License Agreement. The CLA bot will prompt you automatically
when you open your first PR. This is a one-time requirement.

## Governance

DGML is governed as an open standard. A steering committee will be established
over time to guide the evolution of the format, with representation from
contributors and adopters.
