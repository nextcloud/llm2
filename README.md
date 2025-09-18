<!--
  - SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
  - SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Nextcloud Local Large Language Model

[![REUSE status](https://api.reuse.software/badge/github.com/nextcloud/llm2)](https://api.reuse.software/info/github.com/nextcloud/llm2)

![](https://raw.githubusercontent.com/nextcloud/llm2/main/img/Logo.png)

An on-premises text processing backend for the [Nextcloud Assistant](https://github.com/nextcloud/assistant) or any app that uses the [text processing functionality](https://docs.nextcloud.com/server/latest/admin_manual/ai/overview.html#tp-consumer-apps).

This app uses [llama.cpp](https://github.com/abetlen/llama-cpp-python) under the hood and is thus compatible with any open-source model in GGUF format.

## Installation

See [the Nextcloud admin documentation](https://docs.nextcloud.com/server/latest/admin_manual/ai/app_llm2.html) for installation instructions and system requirements.

## Development installation

0. (Optional) [Install Nvidia drivers and CUDA on your host system](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202).

1. Create and activate a Python virtual environment:

   ```sh
   python3 -m venv ./venv && . ./venv/bin/activate
   ```

2. Install dependencies:
    
   ```sh
   poetry install
   ```

3. (Optional) Enable hardware acceleration if your system supports it (check the [`llama.cpp` documentation](https://llama-cpp-python.readthedocs.io/en/latest/) for your accelerator).

4. (Optional) Download any additional desired models into the `models` directory:

   Examples:

   ```sh
   wget -nc -P models https://download.nextcloud.com/server/apps/llm/llama-2-7b-chat-ggml/llama-2-7b-chat.Q4_K_M.gguf
   wget -nc -P models https://download.nextcloud.com/server/apps/llm/leo-hessianai-13B-chat-bilingual-GGUF/leo-hessianai-13b-chat-bilingual.Q4_K_M.gguf
   wget -nc -P models https://huggingface.co/Nextcloud-AI/llm_neuralbeagle_14_7b_gguf/resolve/main/neuralbeagle14-7b.Q4_K_M.gguf
   ```

4. Run the app:

   ```sh
   PYTHONUNBUFFERED=1 APP_HOST=0.0.0.0 APP_ID=llm2 APP_PORT=9081 APP_SECRET=12345 APP_VERSION=<APP_VERSION> NEXTCLOUD_URL=http://nextcloud.local python3 lib/main.py
   ```

5. Register the app with the `manual_install` AppAPI deploy daemon (see AppAPI admin settings in Nextcloud).

   With the [Nextcloud Docker dev environment](https://github.com/juliusknorr/nextcloud-docker-dev), you can just run:

   ```sh
   make register
   ```

   Example if Nextcloud is installed on bare metal instead:

   ```sh
   sudo -u www-data php /var/www/nextcloud/occ app_api:app:unregister llm2 --force
   sudo -u www-data php /var/www/nextcloud/occ app_api:app:register llm2 manual_install --json-info "{\"id\":\"llm2\",\"name\":\"Local large language model\",\"daemon_config_name\":\"manual_install\",\"version\":\"<APP_VERSION>\",\"secret\":\"12345\",\"port\":9081}" --wait-finish
   ```

## Development installation using Docker

> [!NOTE]
> Currently, running the Docker image requires that your host system have CUDA/NVIDIA drivers installed and is equipped with a GPU capable of performing the required tasks.

0. [Install Nvidia drivers and CUDA on your host system](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202) and [install NVIDIA Docker toolkit](https://stackoverflow.com/questions/25185405/using-gpu-from-a-docker-container).

1. Build the Docker image:

   ```sh
   docker build --no-cache -f Dockerfile -t llm2:latest .
   ```

2. Run the Docker image:

   ```sh
   sudo docker run -ti -v /var/run/docker.sock:/var/run/docker.sock -e APP_ID=llm2 -e APP_HOST=0.0.0.0 -e APP_PORT=9081 -e APP_SECRET=12345 -e APP_VERSION=<APP_VERSION> -e NEXTCLOUD_URL='<YOUR_NEXTCLOUD_URL_REACHABLE_FROM_INSIDE_DOCKER>' -e CUDA_VISIBLE_DEVICES=0 -p 9081:9081 --gpus all llm2:latest
   ```

3. Register the app.

   With the [Nextcloud Docker dev environment](https://github.com/juliusknorr/nextcloud-docker-dev), you can just run:

   ```sh
   make register
   ```

   Example if Nextcloud is installed on bare metal instead:

   ```sh
   sudo -u www-data php /var/www/nextcloud/occ app_api:app:unregister llm2 --force
   sudo -u www-data php /var/www/nextcloud/occ app_api:app:register llm2 manual_install --json-info "{\"id\":\"llm2\",\"name\":\"Local large language model\",\"daemon_config_name\":\"manual_install\",\"version\":\"<APP_VERSION>\",\"secret\":\"12345\",\"port\":9081}" --wait-finish
   ```
