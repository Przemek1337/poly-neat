"""Entry-point glue shared by every example's ``main()``.

Running an example directly - ``uv run python -m examples.<pkg>.<mod>`` - should
do the obvious thing: evaluate on the requested device, write artifacts
(including TensorBoard event files), and, so the run can actually be watched,
start a TensorBoard server pointed at those files. That last part is the only
thing this module adds on top of the bare ``run_experiment`` call; it lives here
once so all fourteen examples stay identical.

The benchmark harness deliberately does **not** go through here: it calls
``run_experiment`` with ``artifacts_directory=None`` and writes its own JSON, so
building a results grid never spawns fourteen TensorBoard servers.

CLI flags, on top of ``--cpu`` / ``--gpu`` from :mod:`examples._example_cli`:

* ``--no-tensorboard``   - still write event files, but do not start a server
  (use this for scripted batches and detached ``tmux`` runs, where a blocking
  server would stall the loop),
* ``--tensorboard-port`` - base port for the server (default 6006; the next free
  port is used if it is taken, so several examples can run at once).
"""

from __future__ import annotations

import argparse
import socket
import threading
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from examples._example_cli import add_device_arguments, resolve_device
from examples._experiment import ExperimentReport, print_experiment_report
from polyneat.logging_utils.custom_logger import get_logger

logger = get_logger(__name__)


def _find_free_port(preferred_port: int, maximum_attempts: int = 32) -> int | None:
    """Return the first bindable TCP port at or above ``preferred_port``.

    Args:
        preferred_port: Port to try first, so a run that has the default free
            lands on the well-known 6006 that an ``ssh -L`` tunnel expects.
        maximum_attempts: How many consecutive ports to probe before giving up.

    Returns:
        A port nothing is currently listening on, or ``None`` when every probed
        port was taken.
    """
    for candidate_port in range(preferred_port, preferred_port + maximum_attempts):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe_socket:
            probe_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe_socket.bind(("", candidate_port))
            except OSError:
                continue
            return candidate_port
    return None


def _launch_tensorboard_server(log_directory: Path, preferred_port: int) -> str | None:
    """Start an in-process TensorBoard server watching ``log_directory``.

    TensorBoard is a convenience, never load-bearing: any failure to start it is
    logged as a warning and swallowed so the experiment still runs. The server
    runs on daemon threads inside this process, so it stays up exactly as long as
    the process is kept alive (see :func:`run_example_main`) and needs no
    explicit teardown.

    Args:
        log_directory: Directory the ``TensorBoardLogger`` callback writes event
            files to. Created if missing so the server has something to watch.
        preferred_port: First port to try; the next free one is used if taken.

    Returns:
        The server URL if it started, ``None`` otherwise. The URL is what the
        caller re-logs once the run ends, so it survives the per-generation log
        flood instead of scrolling out of sight at the top.
    """
    log_directory.mkdir(parents=True, exist_ok=True)
    port = _find_free_port(preferred_port)
    if port is None:
        logger.warning(
            "No free port near %d; skipping TensorBoard (event files still written to %s).",
            preferred_port,
            log_directory,
        )
        return None
    try:
        from tensorboard import program

        server = program.TensorBoard()
        server.configure(argv=[None, "--logdir", str(log_directory), "--port", str(port)])
        url = server.launch()
    except Exception as launch_error:  # noqa: BLE001 - TB must never sink a run
        logger.warning(
            "Could not start TensorBoard (%s); event files still written to %s.",
            launch_error,
            log_directory,
        )
        return None
    logger.info("TensorBoard: %s", url)
    return url


def run_example_main(
    run_experiment: Callable[..., ExperimentReport],
    artifacts_directory: Path,
) -> None:
    """Parse the CLI, run one example, and serve its TensorBoard.

    The single ``main()`` body every example delegates to: evaluate on the
    requested device, write artifacts to ``artifacts_directory``, and - unless
    ``--no-tensorboard`` - serve those artifacts on a TensorBoard port for the
    whole run, then hold the server open until interrupted so the final curves
    stay inspectable.

    Args:
        run_experiment: The example's ``run_experiment`` function.
        artifacts_directory: Where this example writes its artifacts; the
            TensorBoard server is pointed at its ``tensorboard`` subdirectory.
    """
    parser = argparse.ArgumentParser(description="PolyNEAT example", allow_abbrev=False)
    add_device_arguments(parser)
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="write event files but do not start a TensorBoard server",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="base port for the TensorBoard server (default: 6006)",
    )
    parsed_arguments = parser.parse_args()
    device = resolve_device(parsed_arguments)

    tensorboard_url: str | None = None
    if not parsed_arguments.no_tensorboard:
        tensorboard_url = _launch_tensorboard_server(
            artifacts_directory / "tensorboard", parsed_arguments.tensorboard_port
        )

    try:
        report = run_experiment(device=device, artifacts_directory=artifacts_directory)
        print_experiment_report(report)
        if tensorboard_url is not None:
            logger.info("TensorBoard still serving at %s - press Ctrl+C to stop.", tensorboard_url)
            threading.Event().wait()
    except KeyboardInterrupt:
        logger.info("Stopping.")
