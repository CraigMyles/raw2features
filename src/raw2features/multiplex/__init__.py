"""Named-channel strategies for applying RGB encoders to multiplex images."""

from .base import (
    BoundMultiplexStrategy,
    MultiplexStrategy,
    PreparedMultiplexStrategy,
)
from .channelwise import ChannelwiseStrategy
from .registry import build_strategy

__all__ = [
    "BoundMultiplexStrategy",
    "ChannelwiseStrategy",
    "MultiplexStrategy",
    "PreparedMultiplexStrategy",
    "build_strategy",
]
