"""Built failures: passing turns changed on purpose, so a judge's TNR can be measured.

The hand-written cases almost always pass, so they can't show how often a judge catches a
failure. Each :class:`Mutation` is a generic change to a turn in OpenAI format that breaks
one metric with a known label. It also lists the metrics it leaves as they were, which keep
the base turn's label: a judge that fails one of those is grading outside its own rubric.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, ClassVar

from evals.calibration.labels import LabelledTurn
from evals.evaluators.turn import Turn

Message = dict[str, Any]
"""One message in OpenAI format."""


class Mutation(ABC):
    """A change that turns a passing turn into one that fails a known metric.

    Attributes:
        name: Mutation name, part of the built item's id.
        breaks: Judge metric whose label becomes ``False``.
        keeps: Judge metrics the change doesn't touch; they keep the base turn's label.
    """

    name: ClassVar[str]
    breaks: ClassVar[str]
    keeps: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def applies_to(self, turn: Turn) -> bool:
        """Return whether the change can be made to this turn.

        Args:
            turn: A passing turn.

        Returns:
            ``True`` if :meth:`apply` would produce a failing turn.
        """

    @abstractmethod
    def apply(self, turn: Turn) -> Turn:
        """Return the changed turn; the given one is left untouched.

        Args:
            turn: A turn for which :meth:`applies_to` is ``True``.

        Returns:
            The changed turn.
        """

    def build(self, base: LabelledTurn) -> LabelledTurn | None:
        """Return the built failure of a passing turn.

        Args:
            base: A labelled turn that passes every label (``labels.base_turns``).

        Returns:
            The built item, labelled ``False`` on :attr:`breaks` and with the base's labels
            on :attr:`keeps`; ``None`` if the change doesn't apply to this turn.
        """
        if not self.applies_to(base.turn):
            return None
        kept = {metric: base.labels[metric] for metric in self.keeps if metric in base.labels}
        return LabelledTurn(
            item_id=f"{base.case_id}+{self.name}",
            case_id=base.case_id,
            source="built",
            turn=self.apply(base.turn),
            labels={**kept, self.breaks: False},
        )


class DropEligibilityCall(Mutation):
    """Remove the eligibility check but keep the answer: a decision without its tool."""

    name = "drop_eligibility_call"
    breaks = "tool_selection"
    keeps = ("argument_quality", "trajectory_efficiency")

    def applies_to(self, turn: Turn) -> bool:
        """Return whether the turn calls an eligibility tool.

        Args:
            turn: A passing turn.

        Returns:
            ``True`` if it has a ``check_*_eligibility`` call.
        """
        return any(_is_eligibility(_name(call)) for call in _calls(turn.messages))

    def apply(self, turn: Turn) -> Turn:
        """Drop every eligibility call, its result, and an AI message left without calls.

        Args:
            turn: A turn with an eligibility call.

        Returns:
            The turn without them; the final answer is unchanged.
        """
        dropped = {call["id"] for call in _calls(turn.messages) if _is_eligibility(_name(call))}
        messages: list[Message] = []
        for message in turn.messages:
            if message.get("role") == "tool" and message.get("tool_call_id") in dropped:
                continue
            calls = message.get("tool_calls")
            if calls:
                kept = [call for call in calls if call["id"] not in dropped]
                if not kept:
                    # Its content only led up to the dropped call (often a thinking block).
                    continue
                message = {**message, "tool_calls": kept}
            messages.append(message)
        return replace(turn, messages=tuple(messages))


class SwapOrderId(Mutation):
    """Point every call at the next order id, so the arguments no longer match the request."""

    name = "swap_order_id"
    breaks = "argument_quality"
    keeps = ("tool_selection", "trajectory_efficiency")

    def applies_to(self, turn: Turn) -> bool:
        """Return whether a call takes an order id.

        Args:
            turn: A passing turn.

        Returns:
            ``True`` if a call has an integer ``order_id``.
        """
        return any(isinstance(_args(call).get("order_id"), int) for call in _calls(turn.messages))

    def apply(self, turn: Turn) -> Turn:
        """Add one to the ``order_id`` of every call.

        The tool results are left as they were: the change is in what the agent asked for.

        Args:
            turn: A turn with an ``order_id`` argument.

        Returns:
            The turn with the changed calls.
        """

        def swap(args: dict[str, Any]) -> dict[str, Any]:
            """Return the arguments with the next order id.

            Args:
                args: A call's arguments.

            Returns:
                The changed arguments, or the same ones without an integer ``order_id``.
            """
            order_id = args.get("order_id")
            return {**args, "order_id": order_id + 1} if isinstance(order_id, int) else args

        return replace(turn, messages=_map_call_args(turn.messages, swap))


class CheckEveryItem(Mutation):
    """Check every item of the order when the customer asked about one."""

    name = "check_every_item"
    breaks = "argument_quality"
    keeps = ("tool_selection", "trajectory_efficiency")

    def applies_to(self, turn: Turn) -> bool:
        """Return whether a call checks some items of an order whose items were looked up.

        Args:
            turn: A passing turn.

        Returns:
            ``True`` if a call's ``order_item_ids`` are fewer than the looked-up items.
        """
        items = _looked_up_item_ids(turn.messages)
        return any(
            "order_item_ids" in _args(call) and set(_args(call)["order_item_ids"]) < set(items)
            for call in _calls(turn.messages)
        )

    def apply(self, turn: Turn) -> Turn:
        """Set every ``order_item_ids`` argument to all the looked-up items.

        Args:
            turn: A turn with a partial ``order_item_ids`` argument.

        Returns:
            The turn with the changed calls.
        """
        items = _looked_up_item_ids(turn.messages)

        def every_item(args: dict[str, Any]) -> dict[str, Any]:
            """Return the arguments with every item of the order.

            Args:
                args: A call's arguments.

            Returns:
                The changed arguments, or the same ones without ``order_item_ids``.
            """
            return {**args, "order_item_ids": items} if "order_item_ids" in args else args

        return replace(turn, messages=_map_call_args(turn.messages, every_item))


class RepeatCall(Mutation):
    """Make the last tool call again with the same arguments, a call nothing needed."""

    name = "repeat_call"
    breaks = "trajectory_efficiency"
    keeps = ("tool_selection", "argument_quality")

    def applies_to(self, turn: Turn) -> bool:
        """Return whether the turn has a tool call.

        Args:
            turn: A passing turn.

        Returns:
            ``True`` if it calls a tool.
        """
        return bool(_calls(turn.messages))

    def apply(self, turn: Turn) -> Turn:
        """Repeat the last tool-calling AI message and its results, with new call ids.

        Args:
            turn: A turn with a tool call.

        Returns:
            The turn with the repeated exchange right after the original one.
        """
        messages = list(turn.messages)
        index = max(i for i, message in enumerate(messages) if message.get("tool_calls"))
        call_message = messages[index]
        ids = {call["id"] for call in call_message["tool_calls"]}
        results = [
            i
            for i, message in enumerate(messages)
            if message.get("role") == "tool" and message.get("tool_call_id") in ids
        ]
        repeat = [
            {
                **call_message,
                "tool_calls": [
                    {**call, "id": f"{call['id']}-repeat"} for call in call_message["tool_calls"]
                ],
            },
            *(
                {**messages[i], "tool_call_id": f"{messages[i]['tool_call_id']}-repeat"}
                for i in results
            ),
        ]
        end = max([index, *results]) + 1
        return replace(turn, messages=tuple(messages[:end] + repeat + messages[end:]))


class ContradictDecision(Mutation):
    """Replace the answer with the opposite of the eligibility decision."""

    name = "contradict_decision"
    breaks = "answer_correctness"
    keeps = ("tool_selection", "argument_quality", "trajectory_efficiency")

    def applies_to(self, turn: Turn) -> bool:
        """Return whether the case has an expected decision and the turn checked it.

        Args:
            turn: A passing turn.

        Returns:
            ``True`` if the reference has ``expected_reason_code`` and its tool was called
            with an ``order_id``.
        """
        return self._order_id(turn) is not None

    def apply(self, turn: Turn) -> Turn:
        """Write an answer that states the opposite decision.

        Args:
            turn: A turn for which :meth:`applies_to` is ``True``.

        Returns:
            The turn with the new answer, also as its last AI message.
        """
        expected = turn.reference["expected_reason_code"]
        verb = "returned" if expected["tool"] == "check_return_eligibility" else "cancelled"
        answer = (
            f"I'm sorry, order {self._order_id(turn)} can no longer be {verb}."
            if expected["code"] == "ELIGIBLE"
            else f"Good news: order {self._order_id(turn)} can still be {verb}."
        )
        return replace(turn, answer=answer, messages=_with_final_answer(turn.messages, answer))

    def _order_id(self, turn: Turn) -> int | None:
        """Return the order the expected eligibility tool was called for.

        Args:
            turn: The turn.

        Returns:
            The ``order_id`` of the last call to the reference's tool, or ``None``.
        """
        expected = turn.reference.get("expected_reason_code")
        if not expected:
            return None
        order_ids = [
            _args(call).get("order_id")
            for call in _calls(turn.messages)
            if _name(call) == expected["tool"]
        ]
        last = order_ids[-1] if order_ids else None
        return last if isinstance(last, int) else None


def all_mutations() -> list[Mutation]:
    """Return one of each mutation.

    Returns:
        The mutations, in the order their built items are listed.
    """
    return [
        DropEligibilityCall(),
        SwapOrderId(),
        CheckEveryItem(),
        RepeatCall(),
        ContradictDecision(),
    ]


def build_failures(bases: Sequence[LabelledTurn]) -> list[LabelledTurn]:
    """Apply every mutation to every base turn it fits.

    Args:
        bases: Passing turns, one per case (``labels.base_turns``).

    Returns:
        The built items, per base in mutation order.
    """
    built = []
    for base in bases:
        for mutation in all_mutations():
            item = mutation.build(base)
            if item is not None:
                built.append(item)
    return built


def _calls(messages: Sequence[Message]) -> list[Message]:
    """Return every tool call of a turn, in order.

    Args:
        messages: The turn in OpenAI format.

    Returns:
        The ``tool_calls`` entries of the AI messages.
    """
    return [call for message in messages for call in message.get("tool_calls") or []]


def _name(call: Message) -> str:
    """Return a tool call's tool name.

    Args:
        call: A ``tool_calls`` entry in OpenAI format.

    Returns:
        The tool name.
    """
    name: str = call["function"]["name"]
    return name


def _args(call: Message) -> dict[str, Any]:
    """Return a tool call's parsed arguments.

    Args:
        call: A ``tool_calls`` entry; its arguments are a JSON string.

    Returns:
        The arguments.
    """
    args: dict[str, Any] = json.loads(call["function"].get("arguments") or "{}")
    return args


def _is_eligibility(name: str) -> bool:
    """Return whether a tool is an eligibility check.

    Args:
        name: Tool name.

    Returns:
        ``True`` for ``check_return_eligibility`` and ``check_cancellation_eligibility``.
    """
    return name.startswith("check_") and name.endswith("_eligibility")


def _map_call_args(
    messages: Sequence[Message], change: Callable[[dict[str, Any]], dict[str, Any]]
) -> tuple[Message, ...]:
    """Return a copy of the turn with every call's arguments changed.

    Args:
        messages: The turn in OpenAI format.
        change: Maps a call's arguments to the new ones.

    Returns:
        The changed turn; the given messages are left untouched.
    """

    def _changed(call: Message) -> Message:
        """Return a copy of one call with changed arguments.

        Args:
            call: A ``tool_calls`` entry.

        Returns:
            The changed copy.
        """
        arguments = json.dumps(change(_args(call)))
        return {**call, "function": {**call["function"], "arguments": arguments}}

    return tuple(
        {**message, "tool_calls": [_changed(call) for call in message["tool_calls"]]}
        if message.get("tool_calls")
        else message
        for message in messages
    )


def _looked_up_item_ids(messages: Sequence[Message]) -> list[int]:
    """Return the item ids of the orders ``get_order_details`` returned in the turn.

    Args:
        messages: The turn in OpenAI format.

    Returns:
        Every ``order_item_id`` in the results, in order. Tool results are always JSON
        (design rule 9); a result without an ``order`` (e.g. ``NOT_FOUND``) adds none.
    """
    detail_calls = {call["id"] for call in _calls(messages) if _name(call) == "get_order_details"}
    ids: list[int] = []
    for message in messages:
        if message.get("role") == "tool" and message.get("tool_call_id") in detail_calls:
            order = json.loads(message.get("content") or "{}").get("order") or {}
            ids += [item["order_item_id"] for item in order.get("items", [])]
    return ids


def _with_final_answer(messages: Sequence[Message], answer: str) -> tuple[Message, ...]:
    """Return a copy of the turn whose final AI message says ``answer``.

    Args:
        messages: The turn in OpenAI format.
        answer: The new final answer.

    Returns:
        The turn with its last message replaced if it is a final AI message, or with a new
        one appended otherwise.
    """
    last = messages[-1] if messages else {}
    if last.get("role") == "assistant" and not last.get("tool_calls"):
        return (*messages[:-1], {**last, "content": answer})
    return (*messages, {"role": "assistant", "content": answer})


__all__ = [
    "CheckEveryItem",
    "ContradictDecision",
    "DropEligibilityCall",
    "Mutation",
    "RepeatCall",
    "SwapOrderId",
    "all_mutations",
    "build_failures",
]
