import os
from mem0 import Memory
from mem0.embeddings.openai import OpenAIEmbedding

# qwen3-embed (vLLM) rejects the `dimensions` parameter that mem0 always sends.
# Send model + input only; the server returns its native 4096-dim vectors.
def _embed_no_dims(self, text, memory_action=None):
    text = text.replace("\n", " ")
    return self.client.embeddings.create(input=[text], model=self.config.model).data[0].embedding

OpenAIEmbedding.embed = _embed_no_dims

CONFIG = {
    "vector_store": {"provider": "qdrant", "config": {
        "host": "127.0.0.1", "port": 6333,
        "collection_name": "agents_memory", "embedding_model_dims": 4096}},
    "embedder": {"provider": "openai", "config": {
        "model": os.environ.get("EMBED_MODEL", "qwen3-embed"),
        "openai_base_url": os.environ["EMBED_BASE_URL"],
        "api_key": "EMPTY", "embedding_dims": 4096}},
    "llm": {"provider": "openai", "config": {
        "model": os.environ["LLM_MODEL_EXECUTOR"],
        "openai_base_url": os.environ["LLM_BASE_URL"],
        "api_key": "EMPTY", "temperature": 0}},
    "history_db_path": "/home/mitac/Multi-Agents/state/mem0_history.db",
}

def get_memory() -> Memory:
    os.makedirs("/home/mitac/Multi-Agents/state", exist_ok=True)
    return Memory.from_config(CONFIG)

