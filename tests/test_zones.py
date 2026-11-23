from visionflow.analytics.zones import Zone
from visionflow.config import ZoneConfig
from visionflow.inference.detector import Track


def make_zone() -> Zone:
    return Zone(ZoneConfig(name="queue", type="queue", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)]))


def track(tid: int, x: float, y: float) -> Track:
    return Track(tid, (x - 10, y - 40, x + 10, y), 0.9, 0)


def test_counts_people_inside():
    zone = make_zone()
    state = zone.update([track(1, 50, 50), track(2, 150, 50)], now=0.0)
    assert state.count == 1


def test_dwell_time_accumulates_and_completes():
    zone = make_zone()
    zone.update([track(1, 50, 50)], now=0.0)
    state = zone.update([track(1, 50, 50)], now=4.0)
    assert state.max_dwell_s == 4.0

    state = zone.update([], now=6.0)
    assert state.count == 0
    assert state.total_visits == 1
    assert state.avg_dwell_s == 6.0
