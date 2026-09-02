import socket
from app_run import bind_port  # thin import shim, see step 3


def test_bind_port_returns_a_listening_socket_on_the_preferred_port():
    # Find a definitely-free port, release it, then ask bind_port for it.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()

    sock = bind_port(free)
    try:
        assert sock.getsockname()[1] == free
        # Already accepting: this is the whole point. Announcing a port that
        # only *will* be listening lets the renderer's first request hit a
        # closed socket.
        client = socket.create_connection(("127.0.0.1", free), timeout=2)
        client.close()
    finally:
        sock.close()


def test_bind_port_falls_back_when_taken():
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    taken = holder.getsockname()[1]
    sock = None
    try:
        sock = bind_port(taken)
        chosen = sock.getsockname()[1]
        assert chosen != taken
        assert isinstance(chosen, int)
    finally:
        if sock is not None:
            sock.close()
        holder.close()
