"""
edge/rssi_tracker.py

RSSITracker is implemented inside edge/bie.py alongside the BIE engine
because the two are tightly coupled (BehavioralClassifier reads from RSSITracker
on every call).

This module re-exports RSSITracker for callers who want to import it directly
without pulling in the full BIE.

Usage:
    from edge.rssi_tracker import RSSITracker
"""

from edge.bie import RSSITracker  # noqa: F401  (re-export)

__all__ = ["RSSITracker"]
