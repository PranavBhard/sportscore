"""
Base Pipeline - Sport-agnostic pipeline orchestration.

Provides the framework for data sync pipelines. Sport-specific
apps subclass BasePipeline and define their own pipeline steps
(e.g., ESPN pull, training data generation, stat computation).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a single pipeline step execution."""
    name: str
    success: bool
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    skipped: bool = False

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class StepDefinition:
    """Definition of a pipeline step."""
    name: str
    fn: Callable
    skip_condition: Optional[Callable[['PipelineContext'], bool]] = None
    description: str = ""
    continue_on_failure: bool = False


@dataclass
class PipelineContext:
    """Shared context passed through pipeline steps."""
    league_id: str
    config: Any = None
    dry_run: bool = False
    verbose: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, StepResult] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    success: bool
    step_results: List[StepResult] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def steps_completed(self) -> List[str]:
        return [sr.name for sr in self.step_results if sr.success and not sr.skipped]

    @property
    def steps_failed(self) -> List[str]:
        return [sr.name for sr in self.step_results if not sr.success and not sr.skipped]

    @property
    def steps_skipped(self) -> List[str]:
        return [sr.name for sr in self.step_results if sr.skipped]

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


def _normalize_step(step) -> StepDefinition:
    """Convert a tuple (name, fn) to StepDefinition for backward compat."""
    if isinstance(step, StepDefinition):
        return step
    name, fn = step
    return StepDefinition(name=name, fn=fn)


class BasePipeline(ABC):
    """
    Abstract pipeline. Sport-specific apps define their own steps.

    Usage:
        class HockeyPipeline(BasePipeline):
            def define_steps(self):
                return [
                    StepDefinition("espn_pull", self.pull_espn_data),
                    StepDefinition("compute_stats", self.compute_advanced_stats,
                                   skip_condition=lambda ctx: ctx.dry_run),
                    ("generate_training", self.generate_training_data),  # tuple also works
                ]

            def pull_espn_data(self, context):
                ...
    """

    def __init__(self, league_id: str, db=None):
        self.league_id = league_id
        self.db = db

    @abstractmethod
    def define_steps(self) -> List[Union[StepDefinition, tuple]]:
        """
        Define pipeline steps as list of StepDefinition or (name, callable) tuples.
        Each callable receives a PipelineContext.
        """
        ...

    def on_step_start(self, step: StepDefinition, index: int, total: int):
        """Hook called before a step runs. Override for custom output."""
        pass

    def on_step_complete(self, step: StepDefinition, result: StepResult, index: int, total: int):
        """Hook called after a step completes. Override for custom output."""
        pass

    def on_step_skip(self, step: StepDefinition, index: int, total: int):
        """Hook called when a step is skipped. Override for custom output."""
        pass

    def run(self, **kwargs) -> PipelineResult:
        """Execute the pipeline steps in order."""
        result = PipelineResult(success=True, started_at=datetime.utcnow())

        context = PipelineContext(
            league_id=self.league_id,
            dry_run=kwargs.pop('dry_run', False),
            verbose=kwargs.pop('verbose', False),
            config=kwargs.pop('config', None),
            extra=kwargs,
        )

        raw_steps = self.define_steps()
        steps = [_normalize_step(s) for s in raw_steps]
        total = len(steps)

        for index, step in enumerate(steps):
            # Check skip condition
            if step.skip_condition and step.skip_condition(context):
                step_result = StepResult(name=step.name, success=True, skipped=True)
                result.step_results.append(step_result)
                context.step_results[step.name] = step_result
                self.on_step_skip(step, index, total)
                continue

            # Run step
            step_result = StepResult(name=step.name, success=False, started_at=datetime.utcnow())
            context.step_results[step.name] = step_result
            self.on_step_start(step, index, total)

            try:
                logger.info(f"[{self.league_id}] Running step: {step.name}")
                step.fn(context)
                step_result.success = True
            except Exception as e:
                logger.error(f"[{self.league_id}] Step '{step.name}' failed: {e}")
                step_result.error = str(e)
                if not step.continue_on_failure:
                    step_result.completed_at = datetime.utcnow()
                    result.step_results.append(step_result)
                    self.on_step_complete(step, step_result, index, total)
                    result.success = False
                    result.error = str(e)
                    break

            step_result.completed_at = datetime.utcnow()
            result.step_results.append(step_result)
            self.on_step_complete(step, step_result, index, total)

        result.completed_at = datetime.utcnow()
        return result
