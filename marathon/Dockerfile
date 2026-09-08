FROM python:3.11-slim-bookworm

# Deliberately small queue writer: no Codex account, Wiphand API/database
# credentials, Docker socket, SSH material, CUDA runtime, or GPU devices.
RUN useradd --create-home --uid 1000 --shell /bin/sh runner

USER runner
WORKDIR /workspace/p40-native-research
