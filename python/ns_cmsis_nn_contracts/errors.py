# SPDX-License-Identifier: LicenseRef-Ambiq-Apollo-SDK
"""Errors of ns_cmsis_nn_contracts. Every failure is an exception; nothing degrades to a default silently."""


class ContractError(ValueError):
    """A contract file exists but cannot be trusted (unreadable, inconsistent, or stale for the checkout)."""


class ContractSchemaError(ContractError):
    """A contract file declares a schema this package does not read."""


class PlannerError(RuntimeError):
    """The host planner library cannot be built, loaded or trusted."""


class PlanRefused(ValueError):
    """The planner refused a layer; ``status`` names why (see arm_nn_plan_status)."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status
