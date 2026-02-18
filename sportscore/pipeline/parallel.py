"""
Parallel execution utilities for pipelines.

Provides reusable patterns for parallel task execution:
- ParallelStepGroup: Run independent callables concurrently
- ParallelItemProcessor: Process a list of items in parallel
- ChunkedParallelProcessor: Process DataFrame in parallel chunks
- ProgressTracker: Thread-safe progress tracking with optional rendering
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a single parallel task."""
    name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at is not None and self.completed_at is not None:
            return self.completed_at - self.started_at
        return None


class ProgressTracker:
    """Thread-safe progress tracker with optional background rendering."""

    def __init__(self, total: int, label: str = "Processing",
                 render_fn: Optional[Callable] = None, render_interval: float = 0.5):
        self.total = total
        self.label = label
        self.render_fn = render_fn
        self.render_interval = render_interval
        self._completed = 0
        self._failed = 0
        self._extra: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._stop_event: Optional[threading.Event] = None
        self._render_thread: Optional[threading.Thread] = None

    def update(self, completed: int = 0, failed: int = 0, **extra):
        with self._lock:
            self._completed += completed
            self._failed += failed
            self._extra.update(extra)

    @property
    def completed(self) -> int:
        with self._lock:
            return self._completed

    @property
    def failed(self) -> int:
        with self._lock:
            return self._failed

    @property
    def percent(self) -> float:
        with self._lock:
            return (self._completed / self.total * 100) if self.total > 0 else 0

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self._start_time

    def start_background_render(self):
        if self.render_fn is None or self._render_thread is not None:
            return
        self._stop_event = threading.Event()

        def _loop():
            while not self._stop_event.is_set():
                self.render_fn(self)
                self._stop_event.wait(self.render_interval)

        self._render_thread = threading.Thread(target=_loop, daemon=True)
        self._render_thread.start()

    def stop_background_render(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._render_thread is not None:
            self._render_thread.join(timeout=1)
            self._render_thread = None
        if self.render_fn is not None:
            self.render_fn(self)


class ParallelStepGroup:
    """
    Run independent callables in parallel.

    Usage:
        group = ParallelStepGroup(tasks=[
            ("injuries", compute_injuries, {}),
            ("elo", compute_elo, {}),
            ("rosters", build_rosters, {}),
        ])
        results = group.execute()
        # results["injuries"].success, results["elo"].result, etc.
    """

    def __init__(self, tasks: List[Tuple[str, Callable, Dict]],
                 max_workers: int = 3,
                 on_complete: Optional[Callable[[str, TaskResult], None]] = None,
                 on_error: Optional[Callable[[str, TaskResult], None]] = None):
        self.tasks = tasks
        self.max_workers = max_workers
        self.on_complete = on_complete
        self.on_error = on_error

    def execute(self) -> Dict[str, TaskResult]:
        results: Dict[str, TaskResult] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for name, fn, kwargs in self.tasks:
                task_result = TaskResult(name=name, success=False, started_at=time.time())
                results[name] = task_result
                futures[executor.submit(fn, **kwargs)] = (name, task_result)

            for future in as_completed(futures):
                name, task_result = futures[future]
                try:
                    task_result.result = future.result()
                    task_result.success = True
                    task_result.completed_at = time.time()
                    if self.on_complete:
                        self.on_complete(name, task_result)
                except Exception as e:
                    task_result.error = str(e)
                    task_result.completed_at = time.time()
                    logger.error(f"Parallel task '{name}' failed: {e}")
                    if self.on_error:
                        self.on_error(name, task_result)

        return results


class ParallelItemProcessor:
    """
    Process a list of items in parallel with progress tracking.

    Usage:
        processor = ParallelItemProcessor(
            items=seasons,
            process_fn=lambda season: pull_espn(season),
            max_workers=4,
        )
        results = processor.execute()
        # [(item, result, exception_or_none), ...]
    """

    def __init__(self, items: list, process_fn: Callable,
                 max_workers: int = 4,
                 progress_callback: Optional[Callable[[int, int, Any], None]] = None,
                 item_label_fn: Optional[Callable] = None):
        self.items = items
        self.process_fn = process_fn
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.item_label_fn = item_label_fn or str

    def execute(self) -> List[Tuple[Any, Any, Optional[Exception]]]:
        results = []
        completed = 0
        total = len(self.items)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_fn, item): item
                for item in self.items
            }

            for future in as_completed(futures):
                item = futures[future]
                completed += 1
                try:
                    result = future.result()
                    results.append((item, result, None))
                except Exception as e:
                    results.append((item, None, e))
                    logger.error(f"Failed processing {self.item_label_fn(item)}: {e}")

                if self.progress_callback:
                    self.progress_callback(completed, total, item)

        return results


class ChunkedParallelProcessor:
    """
    Process a DataFrame in parallel chunks.

    Usage:
        processor = ChunkedParallelProcessor(
            df=games_df,
            process_chunk_fn=generate_features,
            chunk_size=500,
            max_workers=32,
        )
        result_df = processor.execute()
    """

    def __init__(self, df, process_chunk_fn: Callable,
                 chunk_size: int = 500, max_workers: int = 32,
                 progress_callback: Optional[Callable[[int, int], None]] = None):
        self.df = df
        self.process_chunk_fn = process_chunk_fn
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.progress_callback = progress_callback

    def execute(self):
        import pandas as pd

        chunks = [
            self.df.iloc[i:i + self.chunk_size]
            for i in range(0, len(self.df), self.chunk_size)
        ]

        total_chunks = len(chunks)
        completed_chunks = 0
        results = [None] * total_chunks

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_chunk_fn, chunk): idx
                for idx, chunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                idx = futures[future]
                completed_chunks += 1
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Chunk {idx} failed: {e}")
                    results[idx] = pd.DataFrame()

                if self.progress_callback:
                    self.progress_callback(completed_chunks, total_chunks)

        # Filter out None results and concatenate
        valid = [r for r in results if r is not None and len(r) > 0]
        if valid:
            return pd.concat(valid, ignore_index=False)
        return pd.DataFrame()
