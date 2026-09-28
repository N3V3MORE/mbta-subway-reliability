"""Command-line entry point for the whole pipeline.

Usage::

    python -m mbta_ds.cli collect    --days 90
    python -m mbta_ds.cli live       --minutes 10
    python -m mbta_ds.cli clean
    python -m mbta_ds.cli features
    python -m mbta_ds.cli validate
    python -m mbta_ds.cli train
    python -m mbta_ds.cli cluster
    python -m mbta_ds.cli figures
    python -m mbta_ds.cli all        --days 90

Every stage is idempotent and reads from the cache written by the previous one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import pandas as pd

from . import config


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------
def stage_collect(args: argparse.Namespace) -> dict:
    from . import collect_lamp, collect_ridership, collect_weather
    from .http import make_session

    config.ensure_dirs()
    session = make_session()
    summary: dict = {}

    # ------------------------------------------------------------------
    # Resolve ONE analysis window shared by every source. LAMP is the long
    # archive and ridership is the shorter one, so the window ends at the
    # earlier of the two publication fronts.
    # ------------------------------------------------------------------
    index = collect_lamp.fetch_index(session, refresh=args.refresh)
    lamp_first, lamp_last = collect_lamp.coverage(index)

    window_end = lamp_last
    first_limit = lamp_first
    ridership_coverage: tuple | None = None

    if not args.no_ridership:
        try:
            rid_first, rid_last = collect_ridership.coverage(session)
            ridership_coverage = (rid_first, rid_last)
            window_end = min(window_end, rid_last)
            first_limit = max(first_limit, rid_first)
            logging.info("ridership coverage: %s -> %s", rid_first, rid_last)
        except Exception as exc:  # noqa: BLE001 - degrade to delay-only analysis
            logging.warning("could not read ridership coverage (%s); continuing", exc)

    if args.end:
        window_end = min(window_end, _parse_date(args.end))
    window_start = max(
        pd.Timestamp(window_end) - pd.Timedelta(days=args.days - 1), pd.Timestamp(first_limit)
    )
    window_start = window_start.date()
    logging.info(
        "shared analysis window: %s -> %s (%d days)",
        window_start, window_end, (window_end - window_start).days + 1,
    )
    summary["window"] = (window_start, window_end)
    config.save_window(window_start, window_end)

    summary["lamp"] = collect_lamp.collect(
        days=args.days,
        start=window_start,
        end=window_end,
        include_static=not args.skip_static,
        include_alerts=not args.skip_alerts,
    )

    if ridership_coverage is not None:
        try:
            summary["ridership"] = collect_ridership.collect(
                start=window_start, end=window_end, session=session
            )
        except Exception as exc:  # noqa: BLE001 - ridership is enrich-not-essential
            logging.warning("ridership collection failed: %s", exc)
            summary["ridership"] = {"error": str(exc)}

    summary["weather"] = collect_weather.collect(
        days=(window_end - window_start).days + 1,
        end=window_end,
        refresh=args.refresh,
    )

    # Station coordinates come from the V3 API (the LAMP static table has none).
    try:
        from . import collect_v3

        summary["stations"] = len(collect_v3.fetch_stations())
    except Exception as exc:  # noqa: BLE001 - only the map figure depends on this
        logging.warning("could not fetch station coordinates: %s", exc)

    logging.info("collection complete")
    return summary


def stage_live(args: argparse.Namespace) -> dict:
    from . import collect_v3

    return collect_v3.collect(minutes=args.minutes, interval=args.interval)


def stage_clean(args: argparse.Namespace) -> dict:
    from . import clean

    return clean.run(refresh=args.refresh)


def stage_features(args: argparse.Namespace) -> dict:
    from . import features

    return features.run(refresh=args.refresh, quick=args.quick)


def stage_validate(args: argparse.Namespace) -> dict:
    from . import validate

    summary = validate.run()
    if summary["failed"]:
        raise SystemExit(f"data validation failed: {summary['failed']}")
    return summary


def stage_train(args: argparse.Namespace) -> dict:
    from . import model_delay

    return model_delay.run(quick=args.quick, full=getattr(args, "full", False))


def stage_cluster(args: argparse.Namespace) -> dict:
    from . import cluster_stations

    return cluster_stations.run()


def stage_figures(args: argparse.Namespace) -> dict:
    from . import viz

    return viz.run()


def stage_tail(args: argparse.Namespace) -> dict:
    from . import model_tail

    return model_tail.run(quick=args.quick)


def stage_crosscheck(args: argparse.Namespace) -> dict:
    from . import crosscheck

    return crosscheck.run()


def stage_story(args: argparse.Namespace) -> dict:
    from . import story

    return story.build()


def stage_report(args: argparse.Namespace) -> dict:
    from . import report

    return report.build()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mbta_ds.cli",
        description="MBTA subway reliability & ridership analytics pipeline.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("collect", help="download LAMP, ridership and weather data")
    p.add_argument("--days", type=int, default=90,
                   help="service days of history to fetch (default: 90)")
    p.add_argument("--end", type=str, default=None,
                   help="ISO date to end the shared window at (default: latest "
                        "date covered by every source)")
    p.add_argument("--no-ridership", action="store_true",
                   help="skip the ridership source and use the freshest LAMP data")
    p.add_argument("--skip-static", action="store_true", help="skip GTFS static tables")
    p.add_argument("--skip-alerts", action="store_true", help="skip the alerts archive")
    p.add_argument("--refresh", action="store_true", help="ignore caches and re-download")
    p.set_defaults(func=stage_collect)

    p = sub.add_parser("live", help="poll the V3 API and record snapshots")
    p.add_argument("--minutes", type=float, default=10.0, help="how long to poll")
    p.add_argument("--interval", type=float, default=60.0, help="seconds between polls")
    p.set_defaults(func=stage_live)

    p = sub.add_parser("clean", help="build the tidy trip-stop table with delays")
    p.add_argument("--refresh", action="store_true", help="rebuild from raw data")
    p.set_defaults(func=stage_clean)

    p = sub.add_parser("features", help="extract the leakage-safe feature table")
    p.add_argument("--quick", action="store_true", help="subsample for a fast run")
    p.add_argument("--refresh", action="store_true", help="rebuild from the clean table")
    p.set_defaults(func=stage_features)

    p = sub.add_parser("validate", help="run data-validity checks on every source")
    p.set_defaults(func=stage_validate)

    p = sub.add_parser("train", help="track A: delay regression and lateness classification")
    p.add_argument("--quick", action="store_true", help="smaller model set / subsample")
    p.add_argument("--full", action="store_true",
                   help="also fit the costly models (random forest, KNN)")
    p.set_defaults(func=stage_train)

    p = sub.add_parser("tail", help="10+ minute early warning and prediction ranges")
    p.add_argument("--quick", action="store_true", help="one horizon, smaller samples")
    p.set_defaults(func=stage_tail)

    p = sub.add_parser("cluster", help="track B: cluster stations by reliability and demand")
    p.set_defaults(func=stage_cluster)

    p = sub.add_parser("figures", help="render interactive and static figures")
    p.set_defaults(func=stage_figures)

    p = sub.add_parser("crosscheck", help="compare our timestamps with TransitMatters (network)")
    p.set_defaults(func=stage_crosscheck)

    p = sub.add_parser("report", help="write reports/report.html and reports/tables/*.csv")
    p.set_defaults(func=stage_report)

    p = sub.add_parser("story", help="write reports/story.html: the results in plain words and charts")
    p.set_defaults(func=stage_story)

    p = sub.add_parser("all", help="run every stage in order")
    p.add_argument("--days", type=int, default=90, help="service days of history")
    p.add_argument("--quick", action="store_true", help="fast subsampled run")
    p.add_argument("--full", action="store_true",
                   help="include the costly models in the training stage")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.stage == "all":
        args.skip_static = False
        args.skip_alerts = False
        args.no_ridership = False
        args.end = None
        args.refresh = False
        for stage in (stage_collect, stage_clean, stage_features, stage_validate,
                      stage_train, stage_tail, stage_cluster, stage_figures, stage_report,
                      stage_story):
            logging.info("=== %s ===", stage.__name__.replace("stage_", ""))
            stage(args)
        return 0

    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
