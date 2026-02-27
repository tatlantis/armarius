"""
Armarius exceptions.

Replaces silent return None with hard Python exceptions so the calling
pipeline cannot continue past a blocked input.
"""


class SecurityError(Exception):
    """
    Raised when Armarius blocks an execution attempt.

    Triggered by:
      - Unsigned input reaching a @protect-decorated function
      - Invalid or tampered signature (always raised, even with allow_context=True)
      - Unsigned input reaching ShieldedAgentExecutor

    Callers should treat this as a hard stop — the protected function
    did not execute and no side effects occurred.
    """
    pass
