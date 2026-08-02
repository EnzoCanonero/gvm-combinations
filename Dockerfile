FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/EnzoCanonero/gvm-combinations"
LABEL org.opencontainers.image.authors="Enzo Canonero <Enzo.Canonero@rhul.ac.uk>"

WORKDIR /app

COPY pyproject.toml PYPI_README.md LICENSE ./
COPY src/ ./src/

RUN python -m pip install --no-cache-dir . && python -c "import gvm_combinations"

CMD ["python"]
