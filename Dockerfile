FROM python:3.6-slim
WORKDIR /app
COPY . /app
RUN python3 -m pip install -r requirements.txt
EXPOSE 5000
CMD ["python3", "-m", "app.api"]