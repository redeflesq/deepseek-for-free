# Lightweight Python base image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies for the Cloudflare-bypass browser (Chrome via
# DrissionPage/nodriver) plus curl, which entrypoint.sh uses for its
# active-wait health check against the bypass service inside the container.
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    curl \
    unzip \
    xvfb \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | gpg --dearmor -o /usr/share/keyrings/googlechrome-linux-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/googlechrome-linux-keyring.gpg] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package itself via pyproject.toml instead of a bare
# requirements.txt - pinned versions live in pyproject.toml's [project]
# dependencies (see that file's comment: curl_cffi/fastapi/starlette/
# python-multipart already broke this project once on an unpinned upgrade).
# Copying pyproject.toml + src/ before the rest keeps Docker's layer cache
# working the same way requirements.txt-first copying used to.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir .

COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# HEALTHCHECK hits the real application /health (see
# server/routes/health.py's docstring) - not just "port is listening", but
# that DEEPSEEK_AUTH_TOKEN is set and cookies.json actually loaded.
# start-period gives time for the Cloudflare bypass on first boot (bypass
# service + cookie refresh inside entrypoint.sh can take tens of seconds).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -sf http://127.0.0.1:${FASTAPI_SERVER_PORT:-8018}/health | grep -q '"ok"' || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
