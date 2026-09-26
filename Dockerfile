FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ git
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /bin/uv
COPY . /src
RUN uv venv /opt/venv && uv pip install --python /opt/venv --torch-backend cpu --no-sources --no-cache '/src[serve]'

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -r /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENTRYPOINT ["vla-simd-serve"]
