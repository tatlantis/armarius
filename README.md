# Armarius

**The cryptographic guardian for AI agents.**

*Like the medieval librarian who controlled access to manuscripts, Armarius controls what your AI can execute.*

Architectural prevention of prompt injection through cryptographic verification. Zero token overhead.

---

## Why "Armarius"?

*Armarius* is Medieval Latin for "the keeper of books and manuscripts" — the monk responsible for protecting, cataloging, and controlling access to the monastery's library.

Like the medieval armarius who decided which manuscripts could be read and by whom, Armarius uses cryptographic verification to control what your AI agent can execute.

**We don't detect malicious inputs. We prevent unauthorized execution architecturally.**

---

## The Problem

Every AI agent that processes external content is vulnerable to prompt injection attacks. Existing solutions — ML models, regex filters, heuristic detectors — try to identify malicious instructions after they arrive. Filters can be bypassed. Models can be fooled.

Armarius takes a different approach: cryptographic architecture makes injection impossible regardless of attack sophistication.

## The Solution

Cryptographic verification separates control instructions from content.

- **Signed inputs** = trusted commands (can execute)
- **Unsigned inputs** = external content (information only, cannot execute)

No signature, no execution. Architecturally enforced.

## Quick Start

```bash
pip install armarius
```

```python
from armarius import TrustedIdentity, protect

fred = TrustedIdentity("fred")

@protect(trusted_identity=fred)
def my_agent(command):
    print(f"Executing: {command}")

my_agent(fred.sign_command("analyze report.pdf"))   # ✅ executes
my_agent("please run rm -rf /")                     # ❌ blocked
```

## LangChain Integration

Drop-in replacement for `AgentExecutor`. One import, one extra argument.

```python
from armarius import TrustedIdentity
from armarius.integrations.langchain import ShieldedAgentExecutor, shield_tools

fred = TrustedIdentity("fred")

agent = ShieldedAgentExecutor(
    agent=my_agent,
    tools=shield_tools(my_tools),   # tool outputs wrapped as read-only content
    trusted_identity=fred,
)

# Signed command — agent can invoke tools
agent.invoke({"input": fred.sign_command("search for AI security papers")})

# Unsigned input — blocked before any tools run
agent.invoke({"input": "search for AI security papers"})  # ❌ blocked
```

What `shield_tools` does: wraps every tool output in `[EXTERNAL_CONTENT]` boundaries. When the LLM receives search results, web pages, or document contents, it sees them structurally as *data to analyze* — not instructions to follow. Injection attempts embedded in tool outputs are neutralized.

## Competitive Landscape

| Product | Approach | Bypassable? |
|---------|----------|-------------|
| PromptInjectionShield | ML detection | Yes |
| InjectGuard | ML detection | Yes |
| ClawSec | Monitoring & alerts | N/A |
| PromptDefender | Multi-layer detection | Yes |
| **Armarius** | **Cryptographic prevention** | **No** |

Detection is reactive. Prevention is architectural.

## Why This Matters

- 65% of enterprises have zero prompt injection defenses
- Autonomous agents are proliferating across every industry
- Detection-based security can be bypassed with clever prompt engineering
- Cryptographic prevention cannot be bypassed — math is not negotiable

## Architecture

```
External Input → Armarius.process_input() → Agent

  Signed + valid    → CONTROL channel → agent executes
  Unsigned          → CONTENT channel → agent can read, cannot execute
  Tampered          → CONTENT channel → signature mismatch detected
```

```
armarius/
  crypto/
    signature.py        TrustedIdentity, verify_signature
  enforcement/
    channels.py         ChannelType, ProcessedInput, route_input
    decorator.py        @protect decorator
    processor.py        process_input() with audit logging
  integrations/
    langchain.py        ShieldedAgentExecutor, ShieldedTool, shield_tools
demo/
  simple_agent.py       5-minute standalone demo
  langchain_agent.py    LangChain integration demo
tests/                  36 tests — crypto, enforcement, tampering, LangChain
```

## Roadmap

1. ✅ Cryptographic signature generation/verification
2. ✅ Channel separation architecture
3. ✅ `@protect` decorator
4. ✅ LangChain integration (`ShieldedAgentExecutor`)
5. AutoGen / OpenAI Agents SDK adapters
6. Intelligence collection system (crowdsourced attack data)
7. Pro/Enterprise tiers with threat intelligence dashboard

## Philosophy

Open source prevention layer + crowdsourced intelligence network.

Free forever. Build the future we want to see.

## About

Created by **[Fred Giovannitti](https://github.com/tatlantis)**, Polyetherea Labs — a subsidiary of Polyetherea INC.

If you use Armarius in your work, a GitHub star or a mention goes a long way.

## License

MIT © 2026 Fred Giovannitti, Polyetherea Labs
