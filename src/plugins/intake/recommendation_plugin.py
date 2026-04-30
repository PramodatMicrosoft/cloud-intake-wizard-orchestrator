"""
RecommendationPlugin - queries the AI Search decision-tree and JDCP docs indexes
to recommend an environment and the right documentation to the user.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Annotated

from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)


class RecommendationPlugin:
    def __init__(self) -> None:
        self._endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
        self._decision_tree_index = os.environ.get("INTAKE_DECISION_TREE_INDEX", "intake-decision-tree")
        self._docs_index = os.environ.get("INTAKE_DOCS_INDEX", "jdcp-docs")

    async def _search(self, index: str, query: str, top: int = 3) -> list[dict]:
        credential = DefaultAzureCredential()
        client = SearchClient(self._endpoint, index, credential=credential)
        try:
            results = await client.search(
                search_text=query,
                top=top,
                query_type="semantic",
                semantic_configuration_name="semantic-config",
            )
            hits = []
            async for r in results:
                hits.append({k: v for k, v in r.items() if not k.startswith("@")})
            return hits
        finally:
            await client.close()

    @kernel_function(
        description=(
            "Recommend a target cloud environment (Sandbox/Dev/Test/Staging/Prod) given a "
            "free-text description of the workload, classification, and requested environment."
        ),
        name="recommend_environment",
    )
    async def recommend_environment(
        self,
        context: Annotated[str, "Description of the workload context, classification, and any constraints."],
    ) -> str:
        hits = await self._search(self._decision_tree_index, context, top=3)
        return json.dumps({"matches": hits})

    @kernel_function(
        description="Find JDCP documentation pages relevant to the topic the user is asking about.",
        name="recommend_documentation",
    )
    async def recommend_documentation(
        self,
        topic: Annotated[str, "Topic or question (e.g., 'SCAR submission', 'Protected B controls')."],
    ) -> str:
        hits = await self._search(self._docs_index, topic, top=3)
        return json.dumps({"matches": hits})
