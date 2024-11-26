FROM python:3.12-slim
LABEL authors="rd-ml-midterm"

WORKDIR /app
COPY ["./scripts/predict.py", "./datasets/dv_model.pkl", "./Pipfile", "./Pipfile.lock", "./"]
RUN pip install pipenv
RUN pipenv install --system --deploy
EXPOSE 8024
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8024", "predict:app"]