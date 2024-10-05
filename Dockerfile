FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app/

RUN adduser --disabled-password myuser
USER myuser


EXPOSE 8501

CMD ["streamlit","run","app.py","--server.port=8501","--server.enableCORS=false"]