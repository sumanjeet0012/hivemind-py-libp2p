"""Rendezvous interoperability tests (Deliverable 4, Hivemind side).

Topology per test: one rendezvous point (R) + native peers. Pure-Python
(py-libp2p service + client); cross-implementation variants against a
go-libp2p rendezvous point follow the same scenarios.
"""

import pytest
import pytest_asyncio

from hivemind.p2p import P2P
from hivemind.proto import test_pb2

NS = "hivemind-test-rendezvous"
TTL = 120  # py-libp2p minimum registration TTL


@pytest_asyncio.fixture
async def rendezvous_point():
    point = await P2P.create()
    await point.start_rendezvous_server()
    yield point
    await point.shutdown()


@pytest_asyncio.fixture
async def two_peers(rendezvous_point):
    maddrs = await rendezvous_point.get_visible_maddrs()
    peer_a = await P2P.create(initial_peers=maddrs)
    peer_b = await P2P.create(initial_peers=maddrs)
    yield peer_a, peer_b
    await peer_a.shutdown()
    await peer_b.shutdown()


@pytest.mark.asyncio
async def test_register_discover_connect(rendezvous_point, two_peers):
    """E2E: A registers, B discovers A, B runs an RPC on A (discovery-to-connect)."""
    peer_a, peer_b = two_peers
    point_id = rendezvous_point.peer_id

    granted_ttl = await peer_a.rendezvous_register(NS, point_id, ttl=TTL)
    assert granted_ttl >= TTL

    found = await peer_b.rendezvous_discover(NS, point_id)
    by_id = {p.peer_id: p for p in found}
    assert peer_a.peer_id in by_id
    assert len(by_id[peer_a.peer_id].addrs) > 0  # usable addresses came along

    async def square(request, _context):
        return test_pb2.TestResponse(number=request.number**2)

    await peer_a.add_protobuf_handler("square", square, test_pb2.TestRequest)
    # NOTE: peer_b dials purely on discovered state (dial cache fed by discover)
    response = await peer_b.call_protobuf_handler(
        peer_a.peer_id, "square", test_pb2.TestRequest(number=9), test_pb2.TestResponse
    )
    assert response.number == 81


@pytest.mark.asyncio
async def test_unregister_removes_registration(rendezvous_point, two_peers):
    peer_a, peer_b = two_peers
    point_id = rendezvous_point.peer_id

    await peer_a.rendezvous_register(NS, point_id, ttl=TTL)
    assert any(p.peer_id == peer_a.peer_id for p in await peer_b.rendezvous_discover(NS, point_id))

    await peer_a.rendezvous_unregister(NS, point_id)
    assert all(p.peer_id != peer_a.peer_id for p in await peer_b.rendezvous_discover(NS, point_id))


@pytest.mark.asyncio
async def test_multiple_peers_same_namespace(rendezvous_point):
    maddrs = await rendezvous_point.get_visible_maddrs()
    point_id = rendezvous_point.peer_id
    peers = [await P2P.create(initial_peers=maddrs) for _ in range(3)]
    try:
        for i, peer in enumerate(peers):
            await peer.rendezvous_register(f"{NS}-multi", point_id, ttl=TTL)

        found = await peers[0].rendezvous_discover(f"{NS}-multi", point_id)
        found_ids = {p.peer_id for p in found}
        # A peer discovers (at least) the other two; the point may or may not
        # return self-registrations, so assert on the other peers only.
        for peer in peers[1:]:
            assert peer.peer_id in found_ids
    finally:
        for peer in peers:
            await peer.shutdown()


@pytest.mark.asyncio
async def test_namespace_isolation(rendezvous_point, two_peers):
    peer_a, peer_b = two_peers
    point_id = rendezvous_point.peer_id

    await peer_a.rendezvous_register(f"{NS}-one", point_id, ttl=TTL)
    assert await peer_b.rendezvous_discover(f"{NS}-other", point_id) == []


@pytest.mark.asyncio
@pytest.mark.slow
async def test_registration_expires_after_ttl(rendezvous_point):
    """Real TTL expiry: register with the 120s minimum, poll until gone."""
    import asyncio

    maddrs = await rendezvous_point.get_visible_maddrs()
    point_id = rendezvous_point.peer_id
    peer = await P2P.create(initial_peers=maddrs)
    try:
        ns = f"{NS}-expiry"
        await peer.rendezvous_register(ns, point_id, ttl=120)
        assert any(p.peer_id == peer.peer_id for p in await peer.rendezvous_discover(ns, point_id))

        gone = False
        for _ in range(18):  # up to ~3 min
            await asyncio.sleep(10)
            found = await peer.rendezvous_discover(ns, point_id)
            if all(p.peer_id != peer.peer_id for p in found):
                gone = True
                break
        assert gone, "registration did not expire after its TTL"
    finally:
        await peer.shutdown()
