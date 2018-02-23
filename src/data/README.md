# Master program

This code runs on the server side which is configured as the MQTT broker to receive the messages. It creates a web server which updates a website based on the data which is read from the accelerometer, clearly allowing a coach or other support personnel to identify when a player has suffered a trauma. The script contained within `main.py` handles the creation of the web sever and the MQTT server to listen to incoming messages from all of the connected players.

## Structure

```
.
├── algorithms
│   ├── calibration
│   └── model
└── www
    └── web
        ├── landing
        ├── static
        │   ├── css
        │   ├── img
        │   │   └── team
        │   ├── js
        │   │   └── team -> ../../../../../../data/team
        │   ├── scss
        │   └── vendor
        │       ├── bootstrap
        │       │   ├── css
        │       │   └── js
        │       ├── font-awesome
        │       │   ├── css
        │       │   ├── less
        │       │   ├── scss
        │       │   └── webfonts
        │       └── jquery
        └── templates
            └── team -> ../../../../../data/team
```
The `algorithms` foldter contains all the other programs needed to filter out the data or enable connectivity with the web server. The main directory contains `main.py` which combines all the sub programs into the main loop that runs the server and all pooling to the server.

`www` contains all the necessary stuff for the web server to run. `landing` contains the main configuration file for the web server where all URL linking takes place and where the server responds to POST or GET requersts. `static` contains all the static data and libraries that we use to run the web server, the main css bootstrap enhanced template is under `static/css/`. `templates` contains the HTML templates for the websites that we run.

## Building & Running

Make sure that you have all the dependencies:

`pip install -r requierments.txt`

## For Mac OSX

Follow this link: [Tutorial](https://simplifiedthinking.co.uk/2015/10/03/install-mqtt-server/)

Then just:

`python3 main.py "0.0.0.0" 8080 "192.168.0.183" 1883`

to setup the host address and the port on which the server is going to listen. If a webserver other than one being run locally is used, you need to change `0.0.0.0` to correspond to it's address. `192.168.0.183` also needs to be changed to the address of the laptop on the network which is running the MQTT broker. THe web server and the MQTT broker are running in separate threads.

## Details
The `main.py` configures the broker and the web server and initializes postprocessing of the data and calibration of the sensor before analyzing the values. In addition we have implemented a KMeans ML algorithm to do adata filtration and statistically predict the level of impact. Last but not least we are decompressing the data that we have compressed to reduce the network overhead and processing needs for the slave board. In order to keep player data appropriate only to the team using this technology, some encryption is implimented in the form of bit manipulation, making it more difficult to to work out current status of opponents. Future work would include generation encryption functions based on the device's unique ID.
