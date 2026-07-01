from numpy.random import Generator, default_rng


def create_rng(seed: int | None = None) -> Generator:
    return default_rng(seed)
