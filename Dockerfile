FROM python:3.10

WORKDIR /app

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./.dvc/config /app/.dvc/config
COPY ./dvc-remote.json /app/dvc-remote.json

# dvc remote modify myremote --local gdrive_service_account_json_file_path dvc-remote.json
