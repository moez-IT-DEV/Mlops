# MLOps_Training

## In order to setup this project on your local machine :
* ### Requirements :
    - Python installed
    - DVC installed
    - Git installed

* ### Follow the following steps :
    open your terminal in the desired location where you want to place this project and follow these steps :
  * **clone the project.** type the following command in terminal : \
  git clone https://github.com/EXAMPLE/MLOps_Training.git
  * **Create and activate a virtual environment.** type the following command in terminal : 
  ```
  python3 -m venv env && source env/bin/activate
  ```
  * **Install the requirements.** type the following command in terminal :
  ```
    pip install -r requirements.txt
  ```
  * **Install and initialize DVC.** type the following commands in terminal :
  ```
    pip install dvc && dvc init
  ```
  * **Configure DVC locally and set DagsHub storage as the remote.**
  type the following commands in terminal :
  ```
    dvc config set storage.remote dagshub
    dvc config set storage.dagshub.token YOUR_TOKEN
  ```
  * **Get the data from DVC storage.** type the following command in terminal :
  ```
    dvc get
  ```
