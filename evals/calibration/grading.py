"""Grade labelled turns with the judges, several times each, in parallel."""

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from evals.calibration.labels import LabelledTurn, LabelSource
from evals.evaluators.judges.llm_judge import LLMJudgeEvaluator
from evals.settings import eval_settings
from orderpilot.observability import logger


@dataclass(frozen=True, slots=True)
class Grade:
    """A judge's verdicts on one labelled item, for one metric.

    Attributes:
        item_id: The labelled item.
        case_id: Its hand-written case.
        source: Where its label comes from.
        metric: Judge metric.
        label: The expected verdict.
        verdicts: The judge's verdict on each repeat that didn't fail.
        reasonings: The judge's reasoning, one per verdict.
        errors: Repeats where the judge call failed.
    """

    item_id: str
    case_id: str
    source: LabelSource
    metric: str
    label: bool
    verdicts: tuple[bool, ...]
    reasonings: tuple[str, ...]
    errors: int = 0

    @property
    def complete(self) -> bool:
        """Return whether every repeat gave a verdict.

        Returns:
            ``True`` if there are verdicts and no failed judge calls. Only complete grades
            count towards agreement.
        """
        return bool(self.verdicts) and self.errors == 0

    @property
    def verdict(self) -> bool:
        """Return the majority verdict.

        Returns:
            ``True`` if more than half of the verdicts pass. The repeats are odd, so a
            complete grade has no tie.
        """
        return 2 * sum(self.verdicts) > len(self.verdicts)

    @property
    def flipped(self) -> bool:
        """Return whether the verdicts differ across repeats.

        Returns:
            ``True`` if both a pass and a fail were given.
        """
        return len(set(self.verdicts)) > 1

    @property
    def agrees(self) -> bool:
        """Return whether the majority verdict matches the label.

        Returns:
            ``True`` on agreement.
        """
        return self.verdict == self.label


def grade_items(
    items: Sequence[LabelledTurn], judges: Mapping[str, LLMJudgeEvaluator], repeats: int
) -> list[Grade]:
    """Grade every labelled metric of every item ``repeats`` times.

    Items run in parallel, up to ``eval_settings.MAX_CONCURRENCY`` at once. A metric
    without a judge in ``judges`` is not graded.

    Args:
        items: The labelled items.
        judges: Judge metric to its evaluator.
        repeats: Verdicts per item and metric.

    Returns:
        One grade per item and labelled metric, in item order.
    """
    jobs = [(item, metric) for item in items for metric in item.labels if metric in judges]
    logger.info("Grading {} items: {} judge calls", len(jobs), len(jobs) * repeats)
    with ThreadPoolExecutor(max_workers=eval_settings.MAX_CONCURRENCY) as pool:
        futures = [
            pool.submit(_grade, item, metric, judges[metric], repeats) for item, metric in jobs
        ]
        return [future.result() for future in futures]


def _grade(item: LabelledTurn, metric: str, judge: LLMJudgeEvaluator, repeats: int) -> Grade:
    """Ask one judge about one item ``repeats`` times.

    Args:
        item: The labelled item.
        metric: The judge metric.
        judge: Its evaluator.
        repeats: Number of verdicts to ask for.

    Returns:
        The grade; a failed judge call is counted in ``errors`` instead of a verdict.
    """
    verdicts: list[bool] = []
    reasonings: list[str] = []
    errors = 0
    for _ in range(repeats):
        try:
            verdict = judge.grade(item.turn)
        except Exception:  # noqa: BLE001 - one failed judge call must not stop calibration
            logger.exception("Judge {} failed on {}", metric, item.item_id)
            errors += 1
            continue
        verdicts.append(verdict.verdict)
        reasonings.append(verdict.reasoning)
    return Grade(
        item_id=item.item_id,
        case_id=item.case_id,
        source=item.source,
        metric=metric,
        label=item.labels[metric],
        verdicts=tuple(verdicts),
        reasonings=tuple(reasonings),
        errors=errors,
    )


__all__ = ["Grade", "grade_items"]
