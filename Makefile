# SPDX-FileCopyrightText: 2024 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Welcome to Txt2TxtProvider development. Please use \`make <target>\` where <target> is one of"
	@echo " "
	@echo "  Next commands are only for dev environment with nextcloud-docker-dev!"
	@echo "  They should run from the host you are developing on(with activated venv) and not in the container with Nextcloud!"
	@echo "  "
	@echo "  build-push        build image and upload to ghcr.io"
	@echo "  "
	@echo "  run29             install Txt2TxtProvider for Nextcloud 29"
	@echo "  run               install Txt2TxtProvider for Nextcloud Last"
	@echo "  "
	@echo "  For development of this example use PyCharm run configurations. Development is always set for last Nextcloud."
	@echo "  First run 'Txt2TxtProvider' and then 'make registerXX', after that you can use/debug/develop it and easy test."
	@echo "  "
	@echo "  register29        perform registration of running Txt2TxtProvider into the 'manual_install' deploy daemon."
	@echo "  register          perform registration of running Txt2TxtProvider into the 'manual_install' deploy daemon."

.PHONY: build-push
build-push:
	docker login ghcr.io
	docker build --push --platform linux/amd64 --tag ghcr.io/nextcloud/llm2:2.2.0 --tag ghcr.io/nextcloud/llm2:latest .

.PHONY: download-models
download-models:
	cd models && \
	wget -nc https://download.nextcloud.com/server/apps/llm/leo-hessianai-13B-chat-bilingual-GGUF/leo-hessianai-13b-chat-bilingual.Q4_K_M.gguf \
	&& wget -nc https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
	&& wget -nc https://download.nextcloud.com/server/apps/llm/llama-2-7b-chat-ggml/llama-2-7b-chat.Q4_K_M.gguf \
	&& wget -nc https://huggingface.co/Nextcloud-AI/llm_neuralbeagle_14_7b_gguf/resolve/main/neuralbeagle14-7b.Q4_K_M.gguf

.PHONY: run29
run29:
	docker exec master-stable29-1 sudo -u www-data php occ app_api:app:unregister llm2 --silent || true
	docker exec master-stable29-1 sudo -u www-data php occ app_api:app:register llm2 --force-scopes \
		--info-xml https://raw.githubusercontent.com/cloud-py-api/llm2/appinfo/info.xml

.PHONY: run
run:
	docker exec master-nextcloud-1 sudo -u www-data php occ app_api:app:unregister llm2 --silent || true
	docker exec master-nextcloud-1 sudo -u www-data php occ app_api:app:register llm2 --force-scopes \
		--info-xml https://raw.githubusercontent.com/cloud-py-api/llm2/appinfo/info.xml

.PHONY: register29
register29:
	docker exec master-stable29-1 sudo -u www-data php occ app_api:app:unregister llm2 --silent || true
	docker exec master-stable29-1 sudo -u www-data php occ app_api:app:register llm2 manual_install --json-info \
  "{\"id\":\"llm2\",\"name\":\"Local large language model\",\"daemon_config_name\":\"manual_install\",\"version\":\"1.0.0\",\"secret\":\"12345\",\"port\":9081,\"scopes\":[\"AI_PROVIDERS\"],\"system\":0}" \
  --force-scopes --wait-finish

.PHONY: register
register:
	docker exec master-nextcloud-1 sudo -u www-data php occ app_api:app:unregister llm2 --silent || true
	docker exec master-nextcloud-1 sudo -u www-data php occ app_api:app:register llm2 manual_install --json-info \
  "{\"id\":\"llm2\",\"name\":\"Local large language model\",\"daemon_config_name\":\"manual_install\",\"version\":\"1.0.0\",\"secret\":\"12345\",\"port\":9081,\"scopes\":[\"AI_PROVIDERS\"],\"system\":0}" \
  --force-scopes --wait-finish
