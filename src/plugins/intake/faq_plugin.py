"""FaqPlugin - lookups against the FAQ / glossary index."""
from __future__ import annotations

import json
import os
from typing import Annotated

from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from semantic_kernel.functions import kernel_function


class FaqPlugin:
    def __init__(self) -> None:
        self._endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
        self._index = os.environ.get("INTAKE_FAQ_INDEX", "intake-faq")

    @kernel_function(
        description=(
            "Look up an FAQ or glossary term (e.g., 'What is SMTP?', 'What is FPM ID?'). "
            "Returns the top FAQ entries from the JDCP knowledge base."
        ),
        name="lookup",
    )
    async def lookup(
        self,
        question: Annotated[str, "The user's question or term to define."],
        language: Annotated[str, "Two-letter language code (en or fr). Default: en."] = "en",
    ) -> str:
        credential = DefaultAzureCredential()
        client = SearchClient(self._endpoint, self._index, credential=credential)
        try:
            results = await client.search(
                search_text=question,
                top=3,
                filter=f"language eq '{language}'",
            )
            hits = []
            async for r in results:
                hits.append(
                    {
                        "question": r.get("question"),
                        "answer": r.get("answer"),
                        "category": r.get("category"),
                    }
                )
            return json.dumps({"matches": hits})
        finally:
            await client.close()
