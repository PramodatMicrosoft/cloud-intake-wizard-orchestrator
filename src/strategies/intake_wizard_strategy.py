"""
Intake Wizard agent strategy.

Drop into the orchestrator fork at:
  src/strategies/intake_wizard_strategy.py

Then register in:
  - src/strategies/agent_strategies.py     (add `intake_wizard = "intake_wizard"`)
  - src/strategies/agent_strategy_factory.py
        if strategy_type == AgentStrategies.INTAKE_WIZARD:
            return IntakeWizardStrategy()

This strategy uses Semantic Kernel + Azure OpenAI ChatCompletion with tool calling.
The plugins live under src/plugins/intake/.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import AsyncIterable

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory, ChatMessageContent

from .base_agent_strategy import BaseAgentStrategy
from ..plugins.intake.faq_plugin import FaqPlugin
from ..plugins.intake.intake_form_plugin import IntakeFormPlugin
from ..plugins.intake.recommendation_plugin import RecommendationPlugin

logger = logging.getLogger(__name__)


class IntakeWizardStrategy(BaseAgentStrategy):
    """Single-agent strategy that guides users through a JDCP cloud intake submission."""

    AGENT_NAME = "JDCPIntakeAssistant"

    def __init__(self) -> None:
        super().__init__()
        self._system_prompt = self._load_prompt("system_prompt.txt")

    @staticmethod
    def _load_prompt(name: str) -> str:
        prompt_dir = Path(__file__).resolve().parent.parent / "prompts" / "intake_wizard"
        return (prompt_dir / name).read_text(encoding="utf-8")

    def _build_kernel(self, user_id: str) -> Kernel:
        kernel = Kernel()
        kernel.add_service(
            AzureChatCompletion(
                service_id="chat",
                deployment_name=os.environ["CHAT_DEPLOYMENT_NAME"],
                endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-12-01-preview"),
            )
        )
        kernel.add_plugin(IntakeFormPlugin(user_id=user_id), plugin_name="IntakeForm")
        kernel.add_plugin(RecommendationPlugin(), plugin_name="Recommendation")
        kernel.add_plugin(FaqPlugin(), plugin_name="FAQ")
        return kernel

    async def initiate_agent_flow(
        self,
        user_message: str,
        chat_history: ChatHistory,
        *,
        user_id: str,
        conversation_id: str,
        **kwargs,
    ) -> AsyncIterable[ChatMessageContent]:
        kernel = self._build_kernel(user_id=user_id)

        agent = ChatCompletionAgent(
            kernel=kernel,
            name=self.AGENT_NAME,
            instructions=self._system_prompt,
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
        )

        chat_history.add_user_message(user_message)

        async for chunk in agent.invoke_stream(chat_history):
            yield chunk
