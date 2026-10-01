"""MBTA subway reliability & ridership analytics (CS 506 final project).

Two analysis tracks share one dataset builder:

* Track A -- predict subway arrival delay (regression) and P(delay > 5 min)
  (classification) from schedule, delay-propagation, weather and demand
  features; then cluster stations by reliability.
* Track B -- cluster stations by ridership demand profile.

See ``REPORT.md`` for the full write-up and ``cli.py`` for the entry point.
"""

__version__ = "0.1.0"
