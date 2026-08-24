"""A deterministic stand-in model -- and the reason the tests mean anything.

The obvious way to build a fake provider is a lookup table: test case in,
canned JSON out. It makes every test pass and none of them meaningful. A suite
built that way verifies that the table matches the assertions; delete the
retrieval layer, break the prompt renderer, drop the chunk ids, and the table
keeps returning the same answers and the suite stays green. That failure mode
has a name in this project's brief -- "tests that only validate mocks rather
than workflow behavior" -- and it is the specific trap this file exists to
avoid.

So this is not a table. It is a **small, rule-based triage model**: it reads
the prompt it is actually given, and derives its answer from that text and
nothing else. It has no access to the test suite, the evaluation dataset, the
expected outcomes, the database, or the retrieval index. Its entire input is
the same string a real provider would receive.

The consequence is that the workflow is genuinely under test:

* Break retrieval and no excerpts appear in the prompt, so it cannot cite, so
  it answers ``insufficient_evidence`` and the evaluation's citation
  expectations fail.
* Stop rendering chunk ids and every citation it produces is unresolvable, so
  the grounding validator rejects the run.
* Stop passing tool results back into the prompt and it re-requests the same
  lookup, and the "no duplicate tool calls" expectation fails.
* Change the schema without changing the prompt and its output stops
  validating.

None of those are things a canned-response fake could detect.

What it deliberately is *not*: a good classifier. Keyword rules are not a
language model, and the evaluation report is a measure of this pipeline driven
by this stand-in, not a claim about model quality. The same evaluation runs
against a real provider by changing one setting, and that is the number that
would describe the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.ai.pricing import tokens_from_chars
from app.ai.provider import LLMRequest, LLMResponse, TokenUsage

NAME = "fake"
MODEL = "fake-triage-v1"

_ORDER_ID = re.compile(r"\bORD-\d{4,8}\b")
_ACCOUNT_ID = re.compile(r"\bACC-\d{4,8}\b")
#: Currency amounts written the way customers write them.
_AMOUNT = re.compile(
    r"(?:(?P<symbol>[$£€])\s?)?(?P<value>\d{1,6}(?:[.,]\d{2})?)\s*(?P<code>USD|EUR|GBP)?"
)

_SYMBOL_TO_CODE = {"$": "USD", "£": "GBP", "€": "EUR"}

#: Category rules, in priority order. First list with a hit wins, so the more
#: specific categories are listed before the general ones.
_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "spam_or_noise",
        ("unsubscribe", "limited time offer", "crypto investment", "click here to claim"),
    ),
    (
        "account_access",
        (
            "log in",
            "login",
            "sign in",
            "signin",
            "password",
            "locked out",
            "locked",
            "two-factor",
            "two factor",
            "2fa",
            "mfa",
            "authenticator",
            "verification code",
        ),
    ),
    (
        "billing_refund",
        (
            "refund",
            "money back",
            "charged",
            "charge",
            "invoice",
            "billed",
            "overcharged",
            "reimburse",
        ),
    ),
    (
        "shipping_delivery",
        (
            "delivery",
            "deliver",
            "shipping",
            "shipped",
            "parcel",
            "package",
            "tracking",
            "arrive",
            "courier",
        ),
    ),
    (
        "subscription_change",
        (
            "subscription",
            "downgrade",
            "upgrade",
            "cancel my plan",
            "cancel the plan",
            "seats",
            "seat count",
            "plan",
        ),
    ),
    (
        "technical_issue",
        ("error", "crash", "broken", "not working", "bug", "500", "fails to load", "glitch"),
    ),
    ("general_question", ("how do i", "what is", "can you tell me", "question about")),
)

_URGENCY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("critical", ("unauthorised", "unauthorized", "fraud", "stolen", "compromised", "hacked")),
    ("high", ("urgent", "asap", "immediately", "deadline", "blocked", "still waiting")),
    ("low", ("no rush", "just curious", "whenever", "for future reference")),
)

#: Requests this stand-in declines outright. Kept narrow: a refusal must be a
#: judgement about the *request*, not a fallback for anything it cannot answer.
#: Widening this would quietly convert honest "insufficient evidence" answers
#: into refusals, and the two mean completely different things downstream.
_REFUSAL_MARKERS = (
    "someone else's account",
    "someone elses account",
    "without the owner",
    "bypass the verification",
    "bypass verification",
    "skip identity verification",
    "give me the card number",
    "full card number",
)

_TICKET_BLOCK = re.compile(r"<ticket>\s*(?P<body>.*?)\s*</ticket>", re.DOTALL)
_EXCERPT_BLOCK = re.compile(
    r'<excerpt chunk_id="(?P<chunk_id>[^"]+)"[^>]*applies_to="(?P<applies_to>[^"]*)"[^>]*>\s*'
    r"(?P<text>.*?)\s*</excerpt>",
    re.DOTALL,
)
_RESULT_BLOCK = re.compile(
    r'<result index="\d+" tool="(?P<tool>[^"]+)" status="(?P<status>[^"]+)">\s*'
    r"(?P<body>.*?)\s*</result>",
    re.DOTALL,
)
_SENTENCE = re.compile(r"[^.!?]+[.!?]")

_ENTITIES = (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"))


def _unescape(text: str) -> str:
    for needle, replacement in _ENTITIES:
        text = text.replace(needle, replacement)
    return text


@dataclass(frozen=True, slots=True)
class _Excerpt:
    chunk_id: str
    applies_to: str
    text: str


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.casefold()) if len(word) > 2}


class FakeProvider:
    """A rule-based model that answers from the prompt it is handed."""

    name = NAME

    def __init__(self, model: str = MODEL) -> None:
        self._model = model

    def complete(self, request: LLMRequest) -> LLMResponse:
        prompt = "\n".join(message.content for message in request.messages)
        ticket = self._ticket_text(prompt)
        excerpts = self._excerpts(prompt)
        results = self._tool_results(prompt)

        if request.schema_name == "Classification":
            payload = self._classify(ticket)
        elif request.schema_name == "ToolPlan":
            payload = self._plan_tools(ticket, results)
        elif request.schema_name == "Resolution":
            payload = self._resolve(ticket, excerpts, results)
        else:  # pragma: no cover - a new schema without a rule here is a bug
            raise ValueError(f"fake provider has no rule for schema {request.schema_name!r}")

        text = json.dumps(payload, ensure_ascii=False)
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                input_tokens=tokens_from_chars(len(request.system) + len(prompt)),
                output_tokens=tokens_from_chars(len(text)),
                reported=True,
            ),
            model=self._model,
            request_id=None,
        )

    # -- reading the prompt ------------------------------------------------

    @staticmethod
    def _ticket_text(prompt: str) -> str:
        match = _TICKET_BLOCK.search(prompt)
        return _unescape(match.group("body")) if match else ""

    @staticmethod
    def _excerpts(prompt: str) -> list[_Excerpt]:
        return [
            _Excerpt(
                chunk_id=match.group("chunk_id"),
                applies_to=match.group("applies_to"),
                # Unescaped so a quote taken from here matches the source
                # chunk character-for-character. The renderer escapes on the
                # way in; anything reading it has to undo that or every
                # citation from a document containing an angle bracket would
                # fail validation for a reason that has nothing to do with
                # grounding.
                text=_unescape(match.group("text")),
            )
            for match in _EXCERPT_BLOCK.finditer(prompt)
        ]

    @staticmethod
    def _tool_results(prompt: str) -> list[tuple[str, str, str]]:
        return [
            (match.group("tool"), match.group("status"), _unescape(match.group("body")))
            for match in _RESULT_BLOCK.finditer(prompt)
        ]

    # -- step 1: classification -------------------------------------------

    def _classify(self, ticket: str) -> dict[str, object]:
        lowered = ticket.casefold()

        category = "other"
        matched_terms = 0
        for name, keywords in _CATEGORY_RULES:
            hits = sum(1 for keyword in keywords if keyword in lowered)
            if hits:
                category, matched_terms = name, hits
                break

        urgency = "normal"
        for name, keywords in _URGENCY_RULES:
            if any(keyword in lowered for keyword in keywords):
                urgency = name
                break

        # Confidence rises with the number of category terms that matched and
        # is capped below 1.0 -- a keyword matcher that reports certainty would
        # be lying, and this value feeds a policy gate that decides whether a
        # person looks at the ticket.
        confidence = 0.35 if category == "other" else min(0.65 + 0.1 * matched_terms, 0.93)

        order = _ORDER_ID.search(ticket)
        account = _ACCOUNT_ID.search(ticket)
        amount_minor, currency = self._extract_amount(ticket)

        return {
            "category": category,
            "urgency": urgency,
            "confidence": round(confidence, 2),
            "entities": {
                "order_id": order.group(0) if order else None,
                "account_id": account.group(0) if account else None,
                "amount_minor": amount_minor,
                "currency": currency,
                "product": None,
            },
            "summary": self._summarise(ticket),
            "retrieval_query": self._retrieval_query(ticket, category),
        }

    @staticmethod
    def _extract_amount(ticket: str) -> tuple[int | None, str | None]:
        """First money-shaped token, in minor units.

        A bare number with no symbol and no currency code is *not* treated as
        money -- "order 10042" and "$100.42" must not become the same fact.
        """
        for match in _AMOUNT.finditer(ticket):
            symbol, code = match.group("symbol"), match.group("code")
            if not symbol and not code:
                continue
            raw = match.group("value").replace(",", ".")
            minor = int(round(float(raw) * 100)) if "." in raw else int(raw) * 100
            return minor, code or _SYMBOL_TO_CODE.get(symbol or "")
        return None, None

    @staticmethod
    def _summarise(ticket: str) -> str:
        body = ticket.split("\n\n", 1)[-1].strip() or ticket.strip()
        sentence = _SENTENCE.search(body)
        text = (sentence.group(0) if sentence else body).strip()
        return (text[:157] + "...") if len(text) > 160 else text or "Empty message."

    @staticmethod
    def _retrieval_query(ticket: str, category: str) -> str:
        """Category terms plus the ticket's own distinctive words.

        Both halves matter. Category terms alone would retrieve the same
        section for every ticket in a category; ticket words alone would miss
        the vocabulary the policy uses.
        """
        seeds = {
            "billing_refund": "refund window eligible amount charge",
            "shipping_delivery": "delivery late lost parcel tracking dispatch",
            "account_access": "sign in locked password authentication recovery",
            "subscription_change": "subscription upgrade downgrade cancellation proration",
            "technical_issue": "error product not working",
            "general_question": "policy",
            "spam_or_noise": "",
            "other": "",
        }.get(category, "")
        ticket_words = [
            word
            for word in re.findall(r"[A-Za-z]{4,}", ticket)
            if word.casefold()
            not in {"the", "that", "this", "have", "with", "your", "from", "been"}
        ][:12]
        return " ".join(filter(None, [seeds, " ".join(ticket_words)])).strip()

    # -- step 2: tool planning --------------------------------------------

    def _plan_tools(self, ticket: str, results: list[tuple[str, str, str]]) -> dict[str, object]:
        already_called = {tool for tool, _status, _body in results}
        order = _ORDER_ID.search(ticket)
        account = _ACCOUNT_ID.search(ticket)
        lowered = ticket.casefold()

        def plan(tool: str, arguments: dict[str, str], reason: str) -> dict[str, object]:
            return {
                "needs_tool": True,
                "tool_name": tool,
                "arguments_json": json.dumps(arguments),
                "reason": reason,
            }

        if order and "lookup_order" not in already_called:
            return plan(
                "lookup_order",
                {"order_id": order.group(0)},
                "The answer depends on the date, status and condition of this order.",
            )

        # The account id is often only learned *from* the order lookup, which
        # is why this reads the tool results rather than only the ticket. A
        # planner that could see the ticket alone would never discover the
        # plan tier, and every enterprise ticket would escalate as conflicting.
        known_account = account.group(0) if account else self._account_from_results(results)
        if known_account and "lookup_account" not in already_called:
            return plan(
                "lookup_account",
                {"account_id": known_account},
                "Two policy sections apply to different plans; the plan decides which.",
            )

        wants_refund = any(term in lowered for term in ("refund", "money back", "reimburse"))
        if wants_refund and order and "check_refund_eligibility" not in already_called:
            return plan(
                "check_refund_eligibility",
                {"order_id": order.group(0)},
                "The refund window is date arithmetic and should not be estimated.",
            )

        return {
            "needs_tool": False,
            "tool_name": None,
            "arguments_json": None,
            "reason": "The retrieved policy answers this without a further lookup.",
        }

    @staticmethod
    def _account_from_results(results: list[tuple[str, str, str]]) -> str | None:
        for tool, status, body in results:
            if tool == "lookup_order" and status == "ok":
                match = _ACCOUNT_ID.search(body)
                if match:
                    return match.group(0)
        return None

    # -- step 3: resolution ------------------------------------------------

    def _resolve(
        self, ticket: str, excerpts: list[_Excerpt], results: list[tuple[str, str, str]]
    ) -> dict[str, object]:
        lowered = ticket.casefold()

        if any(marker in lowered for marker in _REFUSAL_MARKERS):
            return self._non_answer(
                "refused",
                "This request asks for access or data that cannot be granted through support.",
            )

        if not excerpts:
            return self._non_answer(
                "insufficient_evidence",
                "No retrieved policy section covers this request.",
            )

        ticket_words = _words(ticket)
        ranked = sorted(
            excerpts,
            key=lambda excerpt: (-len(ticket_words & _words(excerpt.text)), excerpt.chunk_id),
        )
        best = ranked[0]
        overlap = len(ticket_words & _words(best.text))
        if overlap < 2:
            # The excerpts came back, but none of them is about this ticket.
            # Answering from the least-irrelevant one is exactly the failure
            # mode retrieval thresholds exist to prevent.
            return self._non_answer(
                "insufficient_evidence",
                "The retrieved sections do not address what the customer asked.",
            )

        quote = self._pick_quote(best.text, ticket_words)
        eligibility = self._eligibility(results)
        category_is_money = any(term in lowered for term in ("refund", "charged", "money back"))

        action: dict[str, object] = {
            "kind": "reply_only",
            "amount_minor": None,
            "target_id": None,
            "justification": (
                "The policy answers the customer's question; no other action is needed."
            ),
        }
        if category_is_money and eligibility is not None and eligibility["eligible"]:
            action = {
                "kind": "issue_refund",
                "amount_minor": int(str(eligibility["max_refundable_minor"])),
                "target_id": str(eligibility["order_id"]),
                "justification": str(eligibility["reason"]),
            }

        risk = (
            "high"
            if action["kind"] == "issue_refund"
            else ("medium" if category_is_money else "low")
        )
        confidence = min(0.55 + 0.06 * overlap, 0.92)

        return {
            "answer_status": "answered",
            "reply_draft": self._draft(quote, action),
            "citations": [{"chunk_id": best.chunk_id, "quote": quote}],
            "recommended_action": action,
            "confidence": round(confidence, 2),
            "risk": risk,
            "escalation_requested": False,
            "escalation_reason": None,
        }

    @staticmethod
    def _non_answer(status: str, reason: str) -> dict[str, object]:
        return {
            "answer_status": status,
            "reply_draft": "",
            "citations": [],
            "recommended_action": {
                "kind": "none",
                "amount_minor": None,
                "target_id": None,
                "justification": reason,
            },
            "confidence": 0.4,
            "risk": "low",
            "escalation_requested": status == "refused",
            "escalation_reason": reason,
        }

    @staticmethod
    def _pick_quote(text: str, ticket_words: set[str]) -> str:
        """The sentence from the excerpt that best matches the ticket.

        Returned exactly as it appears in the source, whitespace and all. That
        is the whole point: a paraphrase here would fail the grounding check,
        which is precisely the behaviour that check is there to enforce.
        """
        sentences: list[str] = [
            str(sentence).strip()
            for sentence in _SENTENCE.findall(text)
            if len(str(sentence).strip()) > 25
        ]
        if not sentences:
            return text.strip()
        return max(
            sentences,
            key=lambda sentence: (len(ticket_words & _words(sentence)), -len(sentence)),
        )

    @staticmethod
    def _eligibility(results: list[tuple[str, str, str]]) -> dict[str, object] | None:
        for tool, status, body in results:
            if tool != "check_refund_eligibility" or status != "ok":
                continue
            parsed: dict[str, object] = {}
            for line in body.splitlines():
                if ": " not in line:
                    continue
                key, _, value = line.partition(": ")
                parsed[key.strip()] = value.strip()
            return {
                "eligible": str(parsed.get("eligible", "False")) == "True",
                "reason": parsed.get("reason", ""),
                "order_id": parsed.get("order_id", ""),
                "max_refundable_minor": int(str(parsed.get("max_refundable_minor", "0"))),
            }
        return None

    @staticmethod
    def _draft(quote: str, action: dict[str, object]) -> str:
        opening = "Thanks for getting in touch."
        if action["kind"] == "issue_refund":
            closing = (
                "We can refund this order. The refund goes back to your original payment "
                "method and usually settles within 5 to 10 business days."
            )
        else:
            closing = "Here is what applies to your situation."
        return f"{opening} {closing}\n\n{quote}"
