# project design
<!-- https://drive.google.com/file/d/16IO84vE3rJWRclbZAnOIEdKAmx5xAi3I/view?usp=sharing -->
![image](https://user-images.githubusercontent.com/40169115/153727141-0e39c6fe-1fdb-42f4-8019-2552bd127751.png)

# bot-detector-ML
this repository is responsible for the machine learning model.
We are currently using two models, a binary classifier and multi label classifier, the binary classifier is responsible for the Real_Player & Unkown_bot classification, if the Real_Player classification is less then 50% the predictions of the multi class classifier are used.

## design

![image](https://media.discordapp.net/attachments/818520902987415602/941709005200437308/unknown.png?width=739&height=676)

![image](https://media.discordapp.net/attachments/818520902987415602/941666319626031124/unknown.png)

## building this repository

### Install:
* [Docker](https://docs.docker.com/get-docker/)
*  [Github desktop](https://desktop.github.com/)
    * [Git windows](https://gitforwindows.org),  [Git unix](https://git-scm.com/download/linux) will also work.
* An integrated development environment (IDE).
    * We recommend [VSCode](https://code.visualstudio.com), but any IDE will work.

### Setup:
1. Open a terminal `cmd`
2. Navigate `cd` to where you want to save our code.
3. The command below will Create a folder `bot-detector` with two sub folders `remote` & `local` & download the remote repositories in the `remote` folder.
    * To add the repositories in github desktop, select `File` on the top left than click `Add local repository`, and navigate to the cloned repositories.

Windows
```
mkdir bot-detector\remote bot-detector\local && cd bot-detector\remote
git clone https://github.com/Bot-detector/Bot-Detector-Core-Files.git
git clone https://github.com/Bot-detector/bot-detector-mysql.git
git clone https://github.com/Bot-detector/bot-detector-ML.git
```
Linux
```
mkdir -p bot-detector/{remote,local}
git clone https://github.com/Bot-detector/Bot-Detector-Core-Files.git
git clone https://github.com/Bot-detector/bot-detector-mysql.git
git clone https://github.com/Bot-detector/bot-detector-ML.git
```

4. Now you can start the project, the command below will create the necessary docker containers, the first time might take a couple minutes. **Make sure docker desktop is running!**
```
cd Bot-Detector-Core-Files
docker-compose up --build
```
5. In the terminal you will now see `/usr/sbin/mysqld: ready for connections.` this means the database is ready.
6. Test the api's: 
    * Core api: ```http://localhost:5000/```
    * Machine learning: ```http://localhost:8000/```

adding /docs at the end will give return the swagger documentation for the components `/docs`
