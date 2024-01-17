# Nextcloud Free Prompt Text Processing Provider

Uses [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) for fast text processing.

*Using the RTX 3060, model is capable of processing one request with **192** tokens in 3.5 seconds.*

> [!WARNING]
> Note: Model loaded in GPU memory uses ~4.3GB.
> Can be run on CPU but will require ~2x of memory.
>
> Models remain in memory after a request, for faster processing of subsequent requests.
>
> Memory is freed after an hour of inactivity.
>

> [!IMPORTANT]
> Only English language is supported

### Only Nextcloud version 29 and higher is supported
