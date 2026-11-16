FROM python:3.12-slim-trixie

RUN apt-get update && apt-get install -y --no-install-recommends git coreutils tar

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
WORKDIR /opt/memory-graph
COPY memory-graph/pyproject.toml memory-graph/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project \
    && mkdir /workspace \
    && git --no-lazy-fetch --version
ENV PATH="/opt/memory-graph/.venv/bin:${PATH}"
RUN python -c "import filelock, jsonschema, ruamel.yaml"
