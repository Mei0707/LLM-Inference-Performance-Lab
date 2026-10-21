FROM python:3.11.14-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/inference-lab-matplotlib
COPY requirements.lock ./
RUN python -m pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --no-build-isolation '.[tokenizer,gpu]'
COPY configs ./configs
ENTRYPOINT ["inference-lab"]
CMD ["--help"]
