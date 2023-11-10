# project-symmetry web version
Project made by IPACT Team.

### Requirements
* Python 3.10

### Instalation
* Open your prefered terminal
* Clone repo to your local machine  
  git clone https://github.com/renata-hafemann/Project-Symmetry-Translation-Models-Optimization.git  
  **(Please note that if you are on the development team, to be able to commit to this remote repository, you need to user the SSH option and not HTTPS. The example above covers the HTTPS option.)**
* Change directory to the folder where you saved the project  
  cd [your project address]
* Install required libraries  
  python pip install -r requirements.txt
* Apply necessary migrations  
  python manage.py migrate
* Start the server  
  python .\manage.py runserver 0.0.0.0:8000
* Access the app in your browser  
  http://127.0.0.1:8000/

Please note the first time the server is started several downloads are made and take some time. Once the app is opened in the browser, the first time a translation is made, the model need to be downloaded, so it'll take a little longer to translate. For each new model used, new downloads are required. The download progress can be checked on the terminal.
    