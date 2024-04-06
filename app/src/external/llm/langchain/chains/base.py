from typing import Any, Dict, Optional

from langchain_core.retrievers import BaseRetriever


class BaseChain:
    def __init__(
        self, llm: Any, retriever: Optional[BaseRetriever] = None
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._intermediates: Dict[str, Any] = {}

    def _save_intermediates(self, value: Any, **kwargs: Any) -> Any:
        key = kwargs.get("key")
        if key is None:
            raise ValueError("Key must be provided to save intermediates.")
        self._intermediates[key] = value
        return value

    def _post_process(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": value, "intermediates": self._intermediates}
