"""End-to-end tests for the standard workflow."""

from ascendmoea import HistoryMonitor, RunConfig, Workflow
from ascendmoea.algorithms import NSGAII
from ascendmoea.problems import DTLZ2


def test_cpu_workflow_runs_and_records_history() -> None:
    """Run a small deterministic optimization with lifecycle monitoring."""
    monitor = HistoryMonitor()
    workflow = Workflow(
        NSGAII(save=10),
        DTLZ2(n=20, m=3, max_fe=80),
        monitors=monitor,
        config=RunConfig(device="cpu", seed=17, cpu_threads=1),
    )
    result = workflow.run()
    assert result.evaluations >= 80
    assert len(result.population) == 20
    assert result.elapsed_seconds > 0
    assert monitor.records
    assert all(record.population.decs.device.type == "cpu" for record in monitor.records)
