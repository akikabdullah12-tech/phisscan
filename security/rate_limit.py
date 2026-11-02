"""
Brute-force protection for the admin login.

Failed attempts are counted per (IP address, username) in a sliding time
window. Once the limit is reached, that pair is locked out, even with the
correct password, until the oldest failure falls out of the window.

The counts are kept in memory. That's fine for a single-process prototype,
but if this were ever run with several worker processes, the counts would
need to move to something shared such as Redis.
"""

import threading
import time
from collections import defaultdict, deque


class LoginRateLimiter:
    def __init__(self, max_attempts: int = 5, window_seconds: int = 900):
        self.max_attempts = max_attempts
        self.window = window_seconds
        self._failures: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> deque:
        failures = self._failures[key]
        while failures and now - failures[0] > self.window:
            failures.popleft()
        return failures

    def is_blocked(self, key: str) -> tuple[bool, int]:
        """(blocked?, seconds until the next attempt is allowed)"""
        now = time.time()
        with self._lock:
            failures = self._recent(key, now)
            if len(failures) >= self.max_attempts:
                return True, int(self.window - (now - failures[0])) + 1
        return False, 0

    def register_failure(self, key: str) -> None:
        with self._lock:
            self._failures[key].append(time.time())

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
