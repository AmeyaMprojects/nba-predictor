"""A short wait for the network after the Mac wakes.

launchd fires a missed job the moment the Mac wakes -- often during a
two-second dark wake, or before Wi-Fi has reconnected after a full wake --
so the first request fails with a DNS error even though the network will
be back seconds later. ``wait_for_network`` polls DNS for the host the job
is about to call, so the job waits that out instead of failing for nothing.
"""

from __future__ import annotations

import socket
import time


def wait_for_network(
    host: str,
    timeout: float = 90,
    interval: float = 5,
    *,
    resolve=socket.getaddrinfo,
    sleep=time.sleep,
    clock=time.monotonic,
) -> bool:
    """True as soon as ``host`` resolves; False once ``timeout`` seconds
    have passed without it resolving. Prints nothing either way -- the
    caller reports a failure in its own words."""
    deadline = clock() + timeout
    while True:
        try:
            resolve(host, 443)
            return True
        except OSError:  # socket.gaierror included
            pass
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        sleep(min(interval, remaining))
