# Nextcloud Local Large Language Model

## Installation
See [the nextcloud admin docs](https://docs.nextcloud.com/server/latest/admin_manual/ai/index.html)

## Development setup

First register a daemon with the `Manual install` template (make sure to set the correct hostname).

```sh
make download-models

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

APP_ID=llm2 APP_PORT=9081 APP_VERSION=1.0.0 APP_SECRET=12345 NEXTCLOUD_URL=http://localhost:8080 python lib/main.py

# Switch to your server installation
./occ app_api:app:unregister llm2
./occ app_api:app:register llm2 manual_install --force-scopes --json-info '{"id":"llm2","name":"Local large language model","daemon_config_name":"manual_install","version":"1.0.0","secret":"12345","port":9081,"scopes":["AI_PROVIDERS", "TASK_PROCESSING"],"system":0}'
```
