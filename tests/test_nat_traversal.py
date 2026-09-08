"""NAT traversal test suite (Deliverable 3, Hivemind side).

Covers, over native py-libp2p without any Go daemon:
- direct connectivity (public/public baseline, both directions)
- Circuit Relay v2 reservation + relayed RPC (relay fallback)
- private-node simulation (no_listen server reachable only via relay)
- NAT failure modes (no reservation, unknown peer, removed handler)

AutoNAT reachability and DCUtR hole-punch flows are tested below. Honesty note:
py-libp2p's AutoNAT server dials requesters back *by peer ID* over the existing
connection rather than by the presented addresses, so on loopback every connected
peer reports reachable. These tests cover the full request/response machinery plus
real negative paths (unknown peers); true unreachability verdicts need real network
topologies (unified-testing).
"""

import pytest
import pytest_asyncio

from hivemind.p2p import P2P, P2PDaemonError
from hivemind.proto import test_pb2


async def _square(request, _context):
    return test_pb2.TestResponse(number=request.number**2)


async def _wait_for(predicate, timeout=10.0):
    import asyncio

    for _ in range(int(timeout / 0.1)):
        if await predicate():
            return True
        await asyncio.sleep(0.1)
    return False


@pytest_asyncio.fixture
async def relay_point():
    relay = await P2P.create()
    await relay.enable_relay(allow_hop=True)
    yield relay
    await relay.shutdown()


@pytest.mark.asyncio
async def test_direct_connectivity_both_directions():
    """Baseline: two public peers, direct dial + RPC each way, no relay involved."""
    server = await P2P.create()
    client = await P2P.create(initial_peers=await server.get_visible_maddrs())
    await client.wait_for_at_least_n_peers(1)

    await server.add_protobuf_handler("sq", _square, test_pb2.TestRequest)
    await client.add_protobuf_handler("sq", _square, test_pb2.TestRequest)

    assert (
        await client.call_protobuf_handler(
            server.peer_id, "sq", test_pb2.TestRequest(number=8), test_pb2.TestResponse
        )
    ).number == 64
    assert (
        await server.call_protobuf_handler(
            client.peer_id, "sq", test_pb2.TestRequest(number=9), test_pb2.TestResponse
        )
    ).number == 81

    await client.shutdown()
    await server.shutdown()


@pytest.mark.asyncio
async def test_relay_reservation(relay_point):
    """A server reserves a slot; the relay point reports it; strangers have none."""
    server = await P2P.create(initial_peers=await relay_point.get_visible_maddrs())

    assert not await relay_point.relay_has_reservation(server.peer_id)
    relay_addr = await server.reserve_relay_slot(relay_point.peer_id)
    assert "/p2p-circuit/p2p/" in relay_addr

    assert await _wait_for(lambda: relay_point.relay_has_reservation(server.peer_id))

    from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID

    stranger = PeerID(b"\x12\x20" + b"\xcd" * 32)
    assert stranger != server.peer_id
    assert not await relay_point.relay_has_reservation(stranger)

    await server.shutdown()


@pytest.mark.asyncio
async def test_relayed_rpc_private_node(relay_point):
    """Private-node simulation: a no_listen server is reachable ONLY via relay."""
    relay_maddrs = await relay_point.get_visible_maddrs()
    server = await P2P.create(no_listen=True, initial_peers=relay_maddrs)
    client = await P2P.create(initial_peers=relay_maddrs)

    # The server advertises nothing dialable directly.
    try:
        await server.get_visible_maddrs()
        raise AssertionError("no_listen server should have no visible maddrs")
    except ValueError:
        pass

    await server.add_protobuf_handler("sq", _square, test_pb2.TestRequest)
    relay_addr = await server.reserve_relay_slot(relay_point.peer_id)
    assert await _wait_for(lambda: relay_point.relay_has_reservation(server.peer_id))

    # Client knows only the relay; it never learns a direct address for the server.
    await client.dial_via_relay(relay_addr)
    response = await client.call_protobuf_handler(
        server.peer_id, "sq", test_pb2.TestRequest(number=7), test_pb2.TestResponse
    )
    assert response.number == 49

    await client.shutdown()
    await server.shutdown()


@pytest.mark.asyncio
async def test_relay_dial_without_reservation_fails(relay_point):
    """Dialing a relay address nobody reserved must fail fast and cleanly."""
    relay_maddrs = await relay_point.get_visible_maddrs()
    server = await P2P.create(initial_peers=relay_maddrs)
    client = await P2P.create(initial_peers=relay_maddrs)

    relay_prefix = str(relay_maddrs[0]).split("/p2p/")[0]
    relay_id = (await relay_point.get_visible_maddrs())[0]
    bogus_addr = f"{relay_prefix}/p2p/{relay_id['p2p']}/p2p-circuit/p2p/{server.peer_id.to_base58()}"
    with pytest.raises(P2PDaemonError):
        await client.dial_via_relay(bogus_addr)

    await client.shutdown()
    await server.shutdown()


@pytest.mark.asyncio
async def test_unknown_peer_dial_fails_cleanly():
    """Dialing a peer ID nobody knows must raise, not hang."""
    from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID

    lonely = await P2P.create()
    ghost = PeerID(b"\x12\x20" + b"\xab" * 32)
    assert ghost != lonely.peer_id
    with pytest.raises(P2PDaemonError):
        await lonely.call_protobuf_handler(
            ghost, "sq", test_pb2.TestRequest(number=1), test_pb2.TestResponse
        )
    await lonely.shutdown()


@pytest.mark.asyncio
async def test_autonat_reachability_check():
    """A asks B to dial it back (AutoNAT); reachable on loopback, status API sane."""
    peer_a = await P2P.create()
    peer_b = await P2P.create(initial_peers=await peer_a.get_visible_maddrs())
    await peer_b.wait_for_at_least_n_peers(1)

    assert await peer_a.autonat_status() in ("unknown", "public", "private")
    assert await peer_a.autonat_check(peer_b.peer_id) is True

    await peer_b.shutdown()
    await peer_a.shutdown()


@pytest.mark.asyncio
async def test_autonat_unknown_server_fails_cleanly():
    """Asking an unreachable peer for a dial-back must raise, not hang."""
    from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID

    lonely = await P2P.create()
    ghost = PeerID(b"\x12\x20" + b"\xee" * 32)
    with pytest.raises(P2PDaemonError):
        await lonely.autonat_check(ghost)
    await lonely.shutdown()


@pytest.mark.asyncio
async def test_dcutr_hole_punch_flow(relay_point):
    """DCUtR negotiation flow: relayed peers upgrade to a direct path.

    On loopback the direct path trivially exists, so this exercises the
    protocol flow (open DCUtR streams, verify direct connection). Real hole
    punching across actual NATs needs network topologies (unified-testing).
    """
    relay_maddrs = await relay_point.get_visible_maddrs()
    peer_a = await P2P.create(initial_peers=relay_maddrs)
    peer_b = await P2P.create(initial_peers=relay_maddrs + await peer_a.get_visible_maddrs())
    await peer_a.enable_dcutr()
    await peer_b.enable_dcutr()

    # Direct connection first: punch must succeed.
    assert await peer_a.hole_punch(peer_b.peer_id) is True

    # Same over a relayed-only path: private node behind the relay.
    private = await P2P.create(no_listen=True, initial_peers=relay_maddrs)
    await private.enable_dcutr()
    relay_addr = await private.reserve_relay_slot(relay_point.peer_id)
    assert await _wait_for(lambda: relay_point.relay_has_reservation(private.peer_id))
    await peer_a.dial_via_relay(relay_addr)
    assert await peer_a.hole_punch(private.peer_id) is True

    await private.shutdown()
    await peer_b.shutdown()
    await peer_a.shutdown()
