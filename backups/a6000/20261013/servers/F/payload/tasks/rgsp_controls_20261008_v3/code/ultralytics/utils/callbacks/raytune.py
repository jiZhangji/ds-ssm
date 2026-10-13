# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.utils import SETTINGS

try:
    assert SETTINGS["raytune"] is True  # verify integration is enabled
    import ray
    from ray import tune
    from ray.air import session

except (ImportError, AssertionError):
    tune = None


def on_fit_epoch_end(trainer):
    """Sends training metrics to Ray Tune at end of each epoch."""
    # Ray changed the private session helper; this optional integration must
    # remain a no-op when training is launched outside Ray Tune.
    session_state = getattr(ray.train._internal, "session", None)
    get_session = getattr(session_state, "get_session", None) if session_state else None
    legacy_get_session = getattr(session_state, "_get_session", None) if session_state else None
    if (get_session and get_session()) or (legacy_get_session and legacy_get_session()):
        metrics = trainer.metrics
        metrics["epoch"] = trainer.epoch
        session.report(metrics)


callbacks = (
    {
        "on_fit_epoch_end": on_fit_epoch_end,
    }
    if tune
    else {}
)
