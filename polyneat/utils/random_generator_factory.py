from numpy.random import Generator, default_rng


def create_seeded_random_generator(seed: int | None = None) -> Generator:
    """Return a numpy ``Generator`` seeded deterministically when ``seed`` is set."""
    return default_rng(seed)
