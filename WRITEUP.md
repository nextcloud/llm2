# Coding Challenge - Issue #241

Total time spent: 4 hours

## 1. Environment Setup

Setting up the environment took significantly longer (approximately 2 hours) than expected due to several issues. These are centered around the fact that I used an outdated source to base myself off when writing the docker compose file and thus was running on an older NextCloud version.

- `host.docker.internal` didn't work, as the Nextcloud container couldn't reach llm2 running on the host, giving the following error: `nc_py_api._exceptions.NextcloudException: [400] Bad Request <request: PUT /ocs/v1.php/apps/app_api/ex-app/status>`, which was difficult to guage the issue from. I eventually managed to fix it by switching to the Docker bridge IP `172.17.0.1` and added it to Nextcloud's `trusted_domains`.
- llm2's task processing provider registration endpoint requires Nextcloud 30+, but my usage of `nextcloud:29` returned `ERROR - Failed to register llama-2-7b-chat.Q4_K_M - core:text2text:summary, Error: [501] <request: POST /ocs/v1.php/apps/app_api/api/v1/ai_provider/task_processing>`. Upgrading fixed this.
- The version of nc_py_api installed by Poetry (`0.24.2`) was out of sync with AppAPI 3.2.3, so I upgraded to `0.30.1`.

After fixing this, llm2 successfully initialized, and upon running a summarization task, I saw the issue to be fixed, being that the progress bar was fixed at 0.00% until the task was completed.

## 2. Investigating the issue

Firstly, I wanted to locate the code that set the progress, so I looked through the source code of `nc_py_api` until I found `set_progress` in the `_TaskProcessingProviderAPI` class, which accepted the task_id and the progress as a float value from 0.00 to 100.00. 

## 3. Pass needed information

I decided to start with the summarization task under `summary.py`. Firstly, in order to call set_progress, we need `SummarizeProcessor` to have access to `nc` and `task_id`, so they were passed as parameters into the constructor function, then `task_processors.py` and `main.py` were modified to support this.

## 4. Investigate how to estimate the response length

Initially, I was thinking about whether the context window `n_ctx` could be used to estimate the response length. However, then I found that inside `task_processors.py`, the model's `max_tokens` can be extracted from the model config, so I extracted that and passed it to `SummarizeProcessor` as another parameter, as this can be used as a more accurate estimation of response length.

## 5. Updating the Progress Bar

The `__call__` method in  `SummarizeProcessor` used `invoke`, which doesn't provide progress. Therefore, I wrote a helped function `_invoke_progress`, which streams the generation and calls `set_progress`, using the max_tokens as the upper bound. For multiple splits, I assumed that each split is roughly equal, so if there are N splits, then split M (1 <= M <= N) would take the progress bar from `(100/N) * (M-1)` to `(100/N) * M`, so the `_invoke_progress` function accepts the current split index and the total number of splits.

## 6. Testing and Limitations

Upon testing, I noticed that the GUI was showing the progress as 0.00% to 1.00%. Therefore, if my code passed 25 to `set_progress`, the GUI would show 0.25%. This seems to be an issue on the side of either the `set_progress` function or the GUI, as there is some kind of division by 100 happening.

One limitation of using the max_tokens as an upper bound is that most responses don't hit this limit or even come close, so it's frequent to see the progress bar jump from a few percentage points to completed or to the boundary of the next split. This upper bound is safe, as responses cannot exceed it, but conservative. Furthermore, this method does not take into account the time it takes to merge the splits at the end, which means that it can stall even while showing 100% progress.