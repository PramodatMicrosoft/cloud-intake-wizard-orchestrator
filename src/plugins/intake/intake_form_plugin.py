"""
IntakeFormPlugin

Persists the user's intake draft and final submission to Cosmos DB.

Containers used (provisioned by infra):
  - intakeRequests   (PK /userId)   - draft + submitted requests
  - userProfiles     (PK /userId)   - role (admin|standard) and display info

Cosmos DB credentials and DB name come from environment variables that the
orchestrator container app receives from App Configuration.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from azure.cosmos.aio import CosmosClient
from azure.identity.aio import DefaultAzureCredential
from semantic_kernel.functions import kernel_function

logger = logging.getLogger(__name__)

# Mandatory-field catalog. Loaded from a packaged copy of intake-fields.json so the
# plugin works even when AI Search is unavailable; AI Search is still queried for
# rich descriptions during conversation.
_FIELDS_FILE = Path(__file__).resolve().parents[3] / "data" / "intake-fields.json"


def _load_field_catalog() -> list[dict[str, Any]]:
    if _FIELDS_FILE.exists():
        return json.loads(_FIELDS_FILE.read_text(encoding="utf-8"))
    logger.warning("intake-fields.json not found at %s; returning empty catalog.", _FIELDS_FILE)
    return []


class IntakeFormPlugin:
    _ENV_TO_FLAG = {
        "Sandbox": "mandatorySandbox",
        "Dev": "mandatoryDev",
        "Test": "mandatoryTest",
        "Staging": "mandatoryStaging",
        "Prod": "mandatoryProd",
    }

    def __init__(self, user_id: str) -> None:
        self._user_id = user_id
        self._catalog = _load_field_catalog()
        self._db_endpoint = os.environ["COSMOS_DB_ENDPOINT"]
        self._db_name = os.environ["COSMOS_DB_DATABASE_NAME"]
        self._container_name = os.environ.get("INTAKE_REQUESTS_CONTAINER", "intakeRequests")

    async def _container(self):
        # New client per call; in production reuse via a singleton in dependencies.py.
        credential = DefaultAzureCredential()
        client = CosmosClient(self._db_endpoint, credential=credential)
        db = client.get_database_client(self._db_name)
        return client, db.get_container_client(self._container_name)

    async def _get_or_create_draft(self) -> dict[str, Any]:
        client, container = await self._container()
        try:
            query = (
                "SELECT TOP 1 * FROM c WHERE c.userId = @uid AND c.status = 'draft' "
                "ORDER BY c._ts DESC"
            )
            params = [{"name": "@uid", "value": self._user_id}]
            items = [
                item async for item in container.query_items(
                    query=query, parameters=params, partition_key=self._user_id
                )
            ]
            if items:
                return items[0]
            draft = {
                "id": str(uuid.uuid4()),
                "userId": self._user_id,
                "status": "draft",
                "fields": {},
                "createdAt": datetime.now(timezone.utc).isoformat(),
            }
            await container.create_item(draft)
            return draft
        finally:
            await client.close()

    @kernel_function(
        description="Return the list of required and good-to-have fields for the chosen environment.",
        name="get_required_fields",
    )
    def get_required_fields(
        self,
        environment: Annotated[str, "One of: Sandbox, Dev, Test, Staging, Prod"],
    ) -> str:
        flag = self._ENV_TO_FLAG.get(environment)
        if not flag:
            return json.dumps({"error": f"Unknown environment '{environment}'"})
        mandatory = [f["fieldName"] for f in self._catalog if f.get(flag)]
        good_to_have = [
            f["fieldName"]
            for f in self._catalog
            if not f.get(flag) and f.get("importance") == "good-to-have"
        ]
        return json.dumps({"mandatory": mandatory, "goodToHave": good_to_have})

    @kernel_function(
        description="Persist a single field value to the user's intake draft.",
        name="update_draft",
    )
    async def update_draft(
        self,
        field: Annotated[str, "Field name (e.g., projectTitle, dataClassification)."],
        value: Annotated[str, "Value to store. For multi-valued fields, use JSON array."],
    ) -> str:
        client, container = await self._container()
        try:
            draft = await self._get_or_create_draft()
            try:
                parsed: Any = json.loads(value)
            except (TypeError, ValueError):
                parsed = value
            draft["fields"][field] = parsed
            draft["updatedAt"] = datetime.now(timezone.utc).isoformat()
            await container.replace_item(draft["id"], draft)
            return json.dumps({"ok": True, "field": field})
        finally:
            await client.close()

    @kernel_function(
        description="Return the user's current draft as JSON.",
        name="get_draft",
    )
    async def get_draft(self) -> str:
        draft = await self._get_or_create_draft()
        return json.dumps(draft)

    @kernel_function(
        description="Finalize the draft and return a tracking ID.",
        name="submit",
    )
    async def submit(self) -> str:
        client, container = await self._container()
        try:
            draft = await self._get_or_create_draft()
            draft["status"] = "submitted"
            draft["submittedAt"] = datetime.now(timezone.utc).isoformat()
            draft["trackingId"] = f"CIW-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{draft['id'][:6].upper()}"
            await container.replace_item(draft["id"], draft)
            return json.dumps({"ok": True, "trackingId": draft["trackingId"]})
        finally:
            await client.close()
