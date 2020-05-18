FROM python:3.6-slim

# copy and install requirements for layer caching
COPY requirements.txt /
RUN python3 -m pip install -r /requirements.txt

# folders with more frequent changes placed at the end to avoid unnecessary rebuilds
COPY app/ /app
WORKDIR /app

EXPOSE 5000
CMD ["python3", "-m", "api"]
