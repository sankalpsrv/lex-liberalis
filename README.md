# Lex-Liberalis - a Semantra fork to search Indian judgments (powered by IndianKanoon)

### Base 

This project is based on the awesome Semantra v0.1.7 which shipped with an MIT License. View Semantra by Dyson Freeman here - github.com/freedmand/semantra

### Example video

https://github.com/sankalpsrv/lex-liberalis/raw/main/LL-1minute.MOV


### Requirements

- API keys
- Python

#### Adding API keys

- You need to add your IndianKanoon API key in the .env file for the app to automatically load your choice of IndianKanoon documents.

*Get your **IndianKanoon API key** by registering for it here - https://api.indiankanoon.org*

- You also need to add your OpenAI keys in the .env file as it is required to create embeddings

*Get your OpenAI API key by paying for them [in your OpenAI account here](https://platform.openai.com/api-keys) this app uses the **ada-v2 model***

#### Python

`version 3.10.12`

`pip install -r requirements.txt`

#### Run command

`python semantra.py`


### How to Contribute

- Make a PR with any of the tasks in the to-do below, or something that you can explain has good reason to be included.
- Make an issue or [email me with suggestions](sankalpsrv.in)

#### To-Do 

- Change source code links and logo-text from "Semantra" to "Lex-Liberalis"
- Reinstall code blocks from Semantra which had custom models' option and add flags option for choosing models
- Remove extra code and requirements

##### License

MIT License
