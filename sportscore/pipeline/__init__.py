"""
Pipeline orchestration infrastructure.

Provides base classes for building sport-specific data pipelines
with step management, parallelization, and progress tracking.
"""

from sportscore.pipeline.base_pipeline import (
    BasePipeline,
    PipelineContext,
    PipelineResult,
    StepDefinition,
    StepResult,
)
from sportscore.pipeline.parallel import (
    ChunkedParallelProcessor,
    ParallelItemProcessor,
    ParallelStepGroup,
    ProgressTracker,
    TaskResult,
)
from sportscore.pipeline.training_csv import TrainingCSVManager

__all__ = [
    "BasePipeline",
    "PipelineContext",
    "PipelineResult",
    "StepDefinition",
    "StepResult",
    "ChunkedParallelProcessor",
    "ParallelItemProcessor",
    "ParallelStepGroup",
    "ProgressTracker",
    "TaskResult",
    "TrainingCSVManager",
]
