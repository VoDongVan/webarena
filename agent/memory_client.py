import requests


class MemoryClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def retrieve(self, query: str, top_k: int = 3) -> str:
        resp = requests.post(
            f"{self.base_url}/retrieve",
            json={"query": query, "top_k": top_k},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        memories = data.get("memories", [])
        if not memories:
            return ""
        lines = []
        for i, m in enumerate(memories, 1):
            lines.append(f"[{i}] {m.get('title', '')}: {m.get('content', '')}")
        return "\n".join(lines)

    def add_memories(self, items: list) -> None:
        payload = [
            {
                "title": item.title,
                "context": item.context,
                "content": item.content,
                "polarity": item.polarity.value if item.polarity else None,
            }
            for item in items
        ]
        resp = requests.post(
            f"{self.base_url}/add_memories",
            json={"memories": payload},
            timeout=10,
        )
        resp.raise_for_status()