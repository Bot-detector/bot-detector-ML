
# api documentation
```sh
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```
# extra info
```sh
POST: to create data.
GET: to read data.
PUT: to update data.
DELETE: to delete data.
```
# keeping fork up to date
```sh
git checkout develop
git pull --rebase upstream develop
git push
```
# setup
## windows
creating a python venv to work in and install the project requirements
```sh
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## linux
```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
# for admin purposes saving & upgrading
when you added some dependancies update the requirements
```sh
venv\Scripts\activate
call pip freeze > requirements.txt
```
when you want to upgrade the dependancies
```sh
venv\Scripts\activate
powershell "(Get-Content requirements.txt) | ForEach-Object { $_ -replace '==', '>=' } | Set-Content requirements.txt"
call pip install -r requirements.txt --upgrade
call pip freeze > requirements.txt
powershell "(Get-Content requirements.txt) | ForEach-Object { $_ -replace '>=', '==' } | Set-Content requirements.txt"
```
# branch cleanup
if your branch gets out of sync and for some reason you have many pushes and pulls, to become insync without pushing some random changes do this
```sh
git fetch https://github.com/Bot-detector/bot-detector-ML.git
``` 