import socket
from app_run import pick_port  # thin import shim, see step 3


def test_pick_port_returns_preferred_when_free():
    # Find a definitely-free port, release it, then ask pick_port for it.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()
    assert pick_port(free) == free


def test_pick_port_falls_back_when_taken():
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    taken = holder.getsockname()[1]
    try:
        chosen = pick_port(taken)
        assert chosen != taken
        assert isinstance(chosen, int)
    finally:
        holder.close()
