<!--
  - SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
  - SPDX-License-Identifier: AGPL-3.0-or-later
-->
# Nextcloud Local Large Language Model

[![REUSE status](https://api.reuse.software/badge/github.com/nextcloud/llm2)](https://api.reuse.software/info/github.com/nextcloud/llm2)

## Installation
See [the nextcloud admin docs](https://docs.nextcloud.com/server/latest/admin_manual/ai/index.html)


## Development installation using docker

**! Requires that your host system have CUDA/NVIDIA drivers installed and is equipped with a GPU capable of performing the required tasks.**

0. [Install Nvidia Drivers an CUDA on your host system](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202) and [install NVIDIA docker toolkit](https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container) 

1. **Build the docker image**

   Example assuming you are in the source directory of the cloned repository

   > docker build --no-cache -f Dockerfile -t llm2:latest .



2. **Run the docker image**

   > sudo docker run -ti -v /var/run/docker.sock:/var/run/docker.sock -e APP_ID=llm2 -e APP_HOST=0.0.0.0 -e APP_PORT=9080 -e APP_SECRET=12345 -e APP_VERSION=1.0.0 -e NEXTCLOUD_URL='<YOUR_NEXTCLOUD_URL_REACHABLE_FROM_INSIDE_DOCKER>' -e CUDA_VISIBLE_DEVICES=0 -p 9080:9080 --gpus all llm2:latest



3. **Register the service**

   Example assuming you are in the source directory of the cloned repository and the docker image of llm2 was successfully build and is up and running

    - *Register manually:*

      > sudo -u www-data php /var/www/nextcloud/occ app_api:app:unregister llm2 --silent || true
      sudo -u www-data php /var/www/nextcloud/occ app_api:app:register llm2 manual_install --json-info "{\"appid\":\"llm2\",\"name\":\"Local large language model\",\"daemon_config_name\":\"manual_install\",\"version\":\"1.0.0\",\"secret\":\"12345\",\"host\":\"localhost\",\"port\":9080,\"scopes\":[\"AI_PROVIDERS\", "TASK_PROCESSING"],\"system_app\":0}" --force-scopes --wait-finish

## Development installation on bare metal

0. [Install Nvidia Drivers an CUDA on your host system](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202)

1. Create a virtual python environment

    > python3 -m venv ./venv && . ./venv/bin/activate

2. Install dependencies (recommended to use a Virtual Environment)
    
    > poetry install

3. If you want hardware acceleration support, check the Llama.cpp docs for your accelerator: https://llama-cpp-python.readthedocs.io/en/latest/

4. Download some models

    > make download-models

4. Run the app

    > PYTHONUNBUFFERED=1 APP_HOST=0.0.0.0 APP_ID=llm2 APP_PORT=9080 APP_SECRET=12345 APP_VERSION=1.1.0 NEXTCLOUD_URL=http://localhost:8080 python3 lib/main.py

5. Register the app with your manual_install AppAPI deploy daemon (see AppAPI admin settings in Nextcloud)

   > sudo -u www-data php /var/www/nextcloud/occ app_api:app:unregister llm2 --force
   sudo -u www-data php /var/www/nextcloud/occ app_api:app:register llm2 manual_install --json-info "{\"appid\":\"llm2\",\"name\":\"Local large language model\",\"daemon_config_name\":\"manual_install\",\"version\":\"1.1.0\",\"secret\":\"12345\",\"host\":\"localhost\",\"port\":9080,\"scopes\":[\"AI_PROVIDERS\"],\"system_app\":0}" --force-scopes --wait-finish
