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
	@echo "  deploy            deploy Txt2TxtProvider to registered 'docker_dev' for Nextcloud Last"
	@echo "  "
	@echo "  run               install Txt2TxtProvider for Nextcloud Last"
	@echo "  "
	@echo "  For development of this example use PyCharm run configurations. Development is always set for last Nextcloud."
	@echo "  First run 'Txt2TxtProvider' and then 'make registerXX', after that you can use/debug/develop it and easy test."
	@echo "  "
	@echo "  register          perform registration of running Txt2TxtProvider into the 'manual_install' deploy daemon."

.PHONY: build-push
build-push:
	docker login ghcr.io
	docker buildx build --push --platform linux/amd64 --tag ghcr.io/nextcloud/llm2:latest .

.PHONY: download-models
download-models:
	cd models
	wget https://download.nextcloud.com/server/apps/llm/leo-hessianai-13B-chat-bilingual-GGUF/leo-hessianai-13b-chat-bilingual.Q4_K_M.gguf
	wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
	wget https://download.nextcloud.com/server/apps/llm/llama-2-7b-chat-ggml/llama-2-7b-chat.Q4_K_M.gguf

.PHONY: deploy
deploy:
	docker exec master-nextcloud-1 sudo -u www-data php occ app_api:app:unregister llm2 --silent || true
	docker exec master-nextcloud-1 sudo -u www-data php occ app_api:app:deploy llm2 docker_dev \
		--info-xml https://raw.githubusercontent.com/cloud-py-api/llm2/appinfo/info.xml

.PHONY: run
run:
	docker exec master-nextcloud-1 sudo -u www-data php occ app_api:app:unregister llm2 --silent || true
	docker exec master-nextcloud-1 sudo -u www-data php occ app_api:app:register llm2 docker_dev --force-scopes \
		--info-xml https://raw.githubusercontent.com/cloud-py-api/llm2/appinfo/info.xml

.PHONY: register
register:
	docker exec master-nextcloud-1 sudo -u www-data php occ app_api:app:unregister llm2 --silent || true
	docker exec master-nextcloud-1 sudo -u www-data php occ app_api:app:register llm2 manual_install --json-info \
  "{\"appid\":\"llm2\",\"name\":\"Local large language model\",\"daemon_config_name\":\"manual_install\",\"version\":\"1.0.0\",\"secret\":\"12345\",\"port\":9081,\"scopes\":[\"AI_PROVIDERS\"],\"system_app\":0}" \
  --force-scopes --wait-finish
