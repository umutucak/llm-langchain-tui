Langchain practice with self-hosted model through ollama using `python3.12.13`

Most constants and hyperparameters are defined in `.env`.

works with ollama, so can install [models](https://ollama.com/library) through `ollama run`. im using the new qwen3.8-27b because why not.

to migrate dev milvus db to prod:

```
-rf milvus_prod.db && cp -r milvus_dev.db milvus_prod.db
```

ollama server systemd env variables:

```
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_MAX_LOADED_MODELS=2"
```

more parameters (inside .env) injected through python
