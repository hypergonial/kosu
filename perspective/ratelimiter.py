# MIT License
#
# Copyright (c) 2022-present HyperGH
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import asyncio
import time
import typing as t
from collections import deque


class RateLimiter:
    def __init__(self, period: float, limit: int) -> None:
        """Rate Limiter implementation for Perspective.

        Parameters
        ----------
        period : float
            The period, in seconds, after which the quota resets.
        limit : int
            The amount of requests allowed in a quota.
        """
        self.period: float = period
        self.limit: int = limit

        self._remaining: int = self.limit  # Remaining queries
        self._reset_at: float = 0.0  # Resets remaining queries if in past

        # deque is basically a list optimized for append and pop at begin&end
        self._queue: t.Deque[asyncio.Event] = deque()
        self._task: t.Optional[asyncio.Task[t.Any]] = None

    @property
    def is_rate_limited(self) -> bool:
        now = time.monotonic()

        if self._reset_at <= now:
            # Reset remaining tokens
            self._remaining = self.limit
            self._reset_at = now + self.period
            return False

        # Return True if we ran out of remaining
        return self._remaining <= 0

    def block(self) -> None:
        """
        Block the ratelimiter for 'period' seconds.
        Called if hitting a 429, this is usually due to improperly configured QPS.
        """
        self._remaining = 0
        self._reset_at = time.monotonic() + self.period

    async def acquire(self) -> None:
        """Acquire a ratelimit, block execution if ratelimited."""
        event = asyncio.Event()

        self._queue.append(event)

        if self._task is None:
            self._task = asyncio.create_task(self._iter_queue())

        await event.wait()

    async def _iter_queue(self) -> None:
        if not self._queue:
            self._task = None
            return

        if self.is_rate_limited:
            # Sleep until ratelimit expires
            await asyncio.sleep(self._reset_at - time.monotonic())

        # Set events while not ratelimited
        while not self.is_rate_limited and self._queue:
            self._remaining -= 1
            self._queue.popleft().set()

        self._task = None
