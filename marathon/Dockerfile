FROM node:20-bookworm-slim

# Deliberately small runner: it has Codex but no Wiphand API/database
# credentials, Docker socket, SSH material, CUDA runtime, or GPU devices.
RUN npm install --global --no-audit --no-fund @openai/codex@0.153.4

USER node
WORKDIR /workspace/p40-native-research
