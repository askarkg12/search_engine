
FROM pytorch/pytorch


WORKDIR /workspace


COPY ./server/requirements.txt /requirements.txt


RUN pip install --no-cache-dir --upgrade -r /requirements.txt


COPY ./server ./server
COPY ./model ./model
# COPY ./weights ./weights
COPY ./utils ./utils
COPY ./dataset ./dataset


CMD ["fastapi", "run", "server/main.py", "--port", "8501"]
