from .scheduler import Scheduler


def get_scheduler_class_from_name(name):
    # Lazy import to improve loading speed and reduce libary dependency.
    if name == "naive":
        from .naive_scheduler import NaiveScheduler

        return NaiveScheduler
    elif name == "outline":
        from .outline_scheduler import OutlineScheduler

        return OutlineScheduler
    elif name == "batch_outline":
        from .outline_batch_scheduler import OutlineBatchScheduler

        return OutlineBatchScheduler
    elif name == "router_batch_outline":
        from .router_outline_batch_scheduler import RouterOutlineBatchScheduler

        return RouterOutlineBatchScheduler
    elif name == "fake_outline":
        from .fake_outline_scheduler import FakeOutlineScheduler

        return FakeOutlineScheduler
    else:
        raise ValueError(f"Unknown scheduler name {name}")


__all__ = ["get_scheduler_class_from_name", "Scheduler"]
