## Code
There are two main piplines in the project: one processes data for catboost model and trains it and the second one does the same with a lightgbm model. \
* Data preperation is done with src/process_data.py. There are some common and different steps for data preperation. The function outputs train test and validation sets.
* Model training is done with the src/train_model.py. The script trains the model saves it, saves model accuracy, ks and auc (being tracked by dvc metrics) and also saves a summary plot for general performance.
* You can run these scripts from cmd, but they need to be run from the main directory (not inside src) and you can specify which pipeline you want by adding a command line argument ("cat" or "lgb")
```
python src/process_data.py "lgb"
python src/train_model.py "lgb"
```
## DVC
* For data storage I'm using a google drive which is connected to a service account. You will need a seperate dvc-remote.json file for the service account credentials which I will send seperatly. 
* Before running the pipeline you can get the main dataset wit:
```
dvc pull data/data.csv
```
* There is a dvc pipline and with it you can run the piplines by simpy running **'dvc repro'**. You specify which pipeline you want ("lgb" or "cat") in the **params.yaml**
## Docker
Run the following command to start the docker container:
```
docker compose up -d
```
After the container is running inside the container you will need to initialize git and also add the dvc-remote.json to dvc file paths:
```
git init
dvc remote modify myremote --local gdrive_service_account_json_file_path dvc-remote.json
```
After this is set up you can run **dvc pull data/data.csv** and **dvc repro**
