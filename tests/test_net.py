"""`wait_for_network`: a short wait for DNS to come back after the Mac
wakes, so a scheduled job that fires the instant the lid opens (before
Wi-Fi reconnects) does not fail for nothing. Everything is injected --
no real DNS lookup, no real sleeping."""

import socket

from predictor.net import wait_for_network


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def test_returns_true_at_once_when_the_host_resolves():
    clock = FakeClock()
    calls = []

    def resolve(host, port):
        calls.append(host)
        return [("ok",)]

    assert wait_for_network("example.test", resolve=resolve, sleep=clock.sleep, clock=clock)
    assert calls == ["example.test"]
    assert clock.sleeps == []


def test_keeps_polling_until_dns_comes_back():
    clock = FakeClock()
    attempts = []

    def resolve(host, port):
        attempts.append(clock.t)
        if len(attempts) < 3:
            raise socket.gaierror(8, "nodename nor servname provided, or not known")
        return [("ok",)]

    assert wait_for_network(
        "example.test", timeout=90, interval=5, resolve=resolve, sleep=clock.sleep, clock=clock
    )
    assert attempts == [0.0, 5.0, 10.0]


def test_gives_up_after_the_timeout():
    clock = FakeClock()

    def resolve(host, port):
        raise socket.gaierror(8, "not known")

    assert not wait_for_network(
        "example.test", timeout=12, interval=5, resolve=resolve, sleep=clock.sleep, clock=clock
    )
    # Never waits past the deadline.
    assert clock.t == 12
    assert clock.sleeps == [5, 5, 2]


def test_any_os_error_from_the_resolver_counts_as_no_network():
    clock = FakeClock()

    def resolve(host, port):
        raise OSError("network is unreachable")

    assert not wait_for_network(
        "example.test", timeout=1, interval=5, resolve=resolve, sleep=clock.sleep, clock=clock
    )
