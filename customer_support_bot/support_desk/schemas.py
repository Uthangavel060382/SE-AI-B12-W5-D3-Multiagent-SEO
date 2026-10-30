"""Structured output for the GraphFlow Manager."""

from typing import Literal

from pydantic import BaseModel, Field


class RoutingDecision(BaseModel):
    """How the Manager routes one employee message. GraphFlow edge conditions read `departments`."""

    departments: list[Literal["IT", "HR", "Admin", "Accounts"]] = Field(
        description="Departments that must act on the latest message; empty when the Manager replies directly"
    )
    priority: Literal["low", "medium", "high", "urgent"]
    reasoning: str = Field(description="One sentence explaining the routing")
    instructions: str = Field(description="What each selected department should do")
    reply_to_employee: str = Field(
        description="Empty when departments is not empty; otherwise the Manager's reply or clarifying question"
    )
