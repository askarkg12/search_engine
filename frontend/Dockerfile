FROM python:3.12-slim

WORKDIR /

COPY ./frontend/requirements.txt /requirements.txt

RUN pip3 install -r requirements.txt

COPY ./frontend ./frontend

EXPOSE 8501


ENTRYPOINT ["streamlit", "run", "frontend/main.py", "--server.port=8501", "--server.address=0.0.0.0"]