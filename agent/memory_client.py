import json
from pathlib import Path

import requests


class MemoryClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        # IDs retrieved so far in the current task (reset each task)
        self._session_retrieved_ids: list[int] = []
        # provenance graph: new_memory_id -> list of parent memory IDs
        self.provenance: dict[int, list[int]] = {}

    def reset_session(self) -> None:
        """Clear per-task retrieved-ID accumulator. Call at the start of each task."""
        self._session_retrieved_ids = []

    @property
    def session_retrieved_ids(self) -> list[int]:
        return list(self._session_retrieved_ids)

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
            mem_id = m.get("id")
            if mem_id is not None:
                self._session_retrieved_ids.append(mem_id)
            lines.append(f"[{i}] {m.get('title', '')}: {m.get('content', '')}")
        return "\n".join(lines)

    def add_memories(self, items: list) -> list[int]:
        """Add memories and record provenance. Returns the assigned IDs."""
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
        assigned_ids: list[int] = resp.json().get("ids", [])
        parent_ids = self.session_retrieved_ids
        for new_id in assigned_ids:
            self.provenance[new_id] = parent_ids
        return assigned_ids

    def save_memories(self, path: str) -> None:
        """Tell the retriever server to persist its memory bank to a JSON file."""
        resp = requests.post(
            f"{self.base_url}/save_memories",
            json={"path": path},
            timeout=30,
        )
        resp.raise_for_status()

    def save_provenance(self, path: str | Path) -> None:
        """Write the provenance graph to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # JSON keys must be strings
        data = {str(k): v for k, v in self.provenance.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved provenance for {len(data)} memories to {path}")