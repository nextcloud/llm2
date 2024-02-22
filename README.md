
# Nextcloud Local Large Language Model

  

**! Requires [`AppAPI`](https://github.com/cloud-py-api/app_api) to work.**

**! Requires that your host system have CUDA/NVIDIA drivers installed and is equipped with a GPU capable of performing the required tasks.**

1. **Build the docker image**

	Example assuming you are in the source directory of the cloned repository

	> docker build --no-cache -f Dockerfile -t llm2:latest .

  

2. **Run the docker image**

	> sudo docker run -ti -v /var/run/docker.sock:/var/run/docker.sock -e APP_ID=llm2 -e APP_HOST=0.0.0.0 -e APP_PORT=9032 -e APP_SECRET=12345 -e APP_VERSION=1.0.0 -e NEXTCLOUD_URL='<YOUR_NEXTCLOUD_URL_REACHABLE_FROM_INSIDE_DOCKER>' -e CUDA_VISIBLE_DEVICES=0 -p 9032:9032 --gpus all llm2:latest

  

3. **Register the service**

	Example assuming you are in the source directory of the cloned repository and the docker image of llm2 was successfully build and is up and running

	> (Hint: In both cases, registering manually or via makefile, adjust the json dictionary that it fits your environment/needs) 

	- *Register manually:*

		> sudo -u www-data php /var/www/nextcloud/occ app_api:app:unregister llm2 --silent || true
	sudo -u www-data php /var/www/nextcloud/occ app_api:app:register llm2 manual_install --json-info "{\"appid\":\"llm2\",\"name\":\"Local large language model\",\"daemon_config_name\":\"manual_install\",\"version\":\"1.0.0\",\"secret\":\"12345\",\"host\":\"192.168.0.199\",\"port\":9032,\"scopes\":[\"AI_PROVIDERS\"],\"system_app\":0}" --force-scopes --wait-finish

	- *Register via Makefile*
		> make register_local
