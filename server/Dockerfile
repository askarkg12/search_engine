
FROM pytorch/pytorch


WORKDIR /workspace


COPY ./server/requirements.txt /requirements.txt


RUN pip install --no-cache-dir --upgrade -r /requirements.txt


COPY ./server ./server
COPY ./model ./model
COPY ./utils ./utils


CMD ["fastapi", "run", "server/main.py", "--port", "80"]