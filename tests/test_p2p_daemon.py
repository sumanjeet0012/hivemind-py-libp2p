import asyncio
import multiprocessing as mp
import os
import tempfile
from contextlib import closing
from functools import partial
from typing import List

import numpy as np
import pytest

from hivemind.p2p import P2P, P2PDaemonError, P2PHandlerError
from hivemind.proto import dht_pb2, test_pb2
from hivemind.utils.multiaddr import Multiaddr
from hivemind.utils.serializer import MSGPackSerializer

from test_utils.networking import get_free_port


async def replicate_if_needed(p2p: P2P, replicate: bool) -> P2P:
    return await P2P.replicate(p2p.daemon_listen_maddr) if replicate else p2p


@pytest.mark.asyncio
async def test_host_stops_on_shutdown():
    p2p = await P2P.create()
    assert p2p.is_alive

    await p2p.shutdown()
    assert not p2p.is_alive
    # Shutdown is idempotent
    await p2p.shutdown()


@pytest.mark.asyncio
async def test_unreachable_bootstrap_peer_is_best_effort():
    # Unlike the Go daemon (which fails fast), the native backend starts
    # successfully and simply has no peers when bootstraps are unreachable.
    p2p = await P2P.create(
        initial_peers=[f"/ip4/127.0.0.1/tcp/{get_free_port()}/p2p/QmdaK4LUeQaKhqSFPRu9N7MvXUEWDxWwtCvPrS444tCgd1"]
    )
    assert await p2p.list_peers() == []
    await p2p.shutdown()


@pytest.mark.asyncio
async def test_identity():
    with tempfile.TemporaryDirectory() as tempdir:
        id1_path = os.path.join(tempdir, "id1")
        id2_path = os.path.join(tempdir, "id2")
        p2ps = await asyncio.gather(*[P2P.create(identity_path=path) for path in [None, None, id1_path, id2_path]])

        # We create the second daemon with id2 separately
        # to avoid a race condition while saving a newly generated identity
        p2ps.append(await P2P.create(identity_path=id2_path))

        # Using the same identity (if any) should lead to the same peer ID
        assert p2ps[-2].peer_id == p2ps[-1].peer_id

        # The rest of peer IDs should be different
        peer_ids = {instance.peer_id for instance in p2ps}
        assert len(peer_ids) == 4

        for instance in p2ps:
            await instance.shutdown()

    with pytest.raises(FileNotFoundError, match=r"The directory.+does not exist"):
        P2P.generate_identity(id1_path)


@pytest.mark.asyncio
async def test_check_if_identity_free():
    with tempfile.TemporaryDirectory() as tempdir:
        id1_path = os.path.join(tempdir, "id1")
        id2_path = os.path.join(tempdir, "id2")

        p2ps = [await P2P.create(identity_path=id1_path)]
        initial_peers = await p2ps[0].get_visible_maddrs()

        p2ps.append(await P2P.create(initial_peers=initial_peers))
        p2ps.append(await P2P.create(initial_peers=initial_peers, identity_path=id2_path))

        with pytest.raises(P2PDaemonError, match=r"Identity.+is already taken by another peer"):
            await P2P.create(initial_peers=initial_peers, identity_path=id1_path)
        # NOTE (native backend, D2): collision detection currently covers peers
        # directly reachable from initial_peers. Swarm-wide detection via peer
        # routing arrives with Rendezvous/DHT integration (Phase 3), so here we
        # check against the identity holder's addresses directly.
        with pytest.raises(P2PDaemonError, match=r"Identity.+is already taken by another peer"):
            await P2P.create(
                initial_peers=await p2ps[-1].get_visible_maddrs(), identity_path=id2_path
            )

        # Must work if a P2P with a certain identity is restarted
        await p2ps[-1].shutdown()
        p2ps.pop()
        p2ps.append(await P2P.create(initial_peers=initial_peers, identity_path=id2_path))

        for instance in p2ps:
            await instance.shutdown()


@pytest.mark.parametrize(
    "host_maddrs",
    [
        [Multiaddr("/ip4/127.0.0.1/tcp/0")],
        pytest.param(
            [Multiaddr("/ip4/127.0.0.1/udp/0/quic-v1")], marks=pytest.mark.skip("quic-v1 is not supported in CI")
        ),
        [Multiaddr("/ip4/127.0.0.1/tcp/0"), Multiaddr("/ip4/127.0.0.1/udp/0/quic")],
    ],
)
@pytest.mark.asyncio
async def test_transports(host_maddrs: List[Multiaddr]):
    server = await P2P.create(host_maddrs=host_maddrs)
    peers = await server.list_peers()
    assert len(peers) == 0

    client = await P2P.create(host_maddrs=host_maddrs, initial_peers=await server.get_visible_maddrs())
    await client.wait_for_at_least_n_peers(1)

    peers = await client.list_peers()
    assert len({p.peer_id for p in peers}) == 1
    peers = await server.list_peers()
    assert len({p.peer_id for p in peers}) == 1


@pytest.mark.asyncio
async def test_replica_shutdown_does_not_affect_primary():
    p2p = await P2P.create()
    p2p_replica = await P2P.replicate(p2p.daemon_listen_maddr)
    assert p2p_replica.peer_id == p2p.peer_id

    await p2p_replica.shutdown()
    assert p2p.is_alive

    await p2p.shutdown()
    assert not p2p.is_alive


@pytest.mark.asyncio
async def test_unary_handler_edge_cases():
    p2p = await P2P.create()
    p2p_replica = await P2P.replicate(p2p.daemon_listen_maddr)

    async def square_handler(data: test_pb2.TestRequest, context):
        return test_pb2.TestResponse(number=data.number**2)

    await p2p.add_protobuf_handler("square", square_handler, test_pb2.TestRequest)

    # try adding a duplicate handler
    with pytest.raises(P2PDaemonError):
        await p2p.add_protobuf_handler("square", square_handler, test_pb2.TestRequest)

    # try adding a duplicate handler from replicated p2p
    with pytest.raises(P2PDaemonError):
        await p2p_replica.add_protobuf_handler("square", square_handler, test_pb2.TestRequest)

    # try dialing yourself
    with pytest.raises(P2PDaemonError):
        await p2p_replica.call_protobuf_handler(
            p2p.peer_id, "square", test_pb2.TestRequest(number=41), test_pb2.TestResponse
        )


@pytest.mark.parametrize(
    "should_cancel,replicate",
    [
        (True, False),
        (True, True),
        (False, False),
        (False, True),
    ],
)
@pytest.mark.asyncio
async def test_call_protobuf_handler(should_cancel, replicate, handle_name="handle"):
    handler_cancelled = False
    server_primary = await P2P.create()
    server = await replicate_if_needed(server_primary, replicate)

    async def ping_handler(request, context):
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            nonlocal handler_cancelled
            handler_cancelled = True
        return dht_pb2.PingResponse(peer=dht_pb2.NodeInfo(node_id=server.peer_id.to_bytes()), available=True)

    await server.add_protobuf_handler(handle_name, ping_handler, dht_pb2.PingRequest)

    client_primary = await P2P.create(initial_peers=await server.get_visible_maddrs())
    client = await replicate_if_needed(client_primary, replicate)
    await client.wait_for_at_least_n_peers(1)

    ping_request = dht_pb2.PingRequest(peer=dht_pb2.NodeInfo(node_id=client.peer_id.to_bytes()), validate=True)
    expected_response = dht_pb2.PingResponse(peer=dht_pb2.NodeInfo(node_id=server.peer_id.to_bytes()), available=True)

    if should_cancel:
        call_task = asyncio.create_task(
            client.call_protobuf_handler(server.peer_id, handle_name, ping_request, dht_pb2.PingResponse)
        )
        await asyncio.sleep(0.25)

        call_task.cancel()

        await asyncio.sleep(0.25)
        assert handler_cancelled
    else:
        actual_response = await client.call_protobuf_handler(
            server.peer_id, handle_name, ping_request, dht_pb2.PingResponse
        )
        assert actual_response == expected_response
        assert not handler_cancelled

    await server.shutdown()
    await server_primary.shutdown()

    await client_primary.shutdown()


@pytest.mark.asyncio
async def test_call_protobuf_handler_error(handle_name="handle"):
    async def error_handler(request, context):
        raise ValueError("boom")

    server = await P2P.create()
    await server.add_protobuf_handler(handle_name, error_handler, dht_pb2.PingRequest)

    client = await P2P.create(initial_peers=await server.get_visible_maddrs())
    await client.wait_for_at_least_n_peers(1)

    ping_request = dht_pb2.PingRequest(peer=dht_pb2.NodeInfo(node_id=client.peer_id.to_bytes()), validate=True)

    with pytest.raises(P2PHandlerError) as excinfo:
        await client.call_protobuf_handler(server.peer_id, handle_name, ping_request, dht_pb2.PingResponse)
    assert "boom" in str(excinfo.value)

    await server.shutdown()
    await client.shutdown()


async def handle_square_stream(_, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    with closing(writer):
        while True:
            try:
                x = MSGPackSerializer.loads(await P2P.receive_raw_data(reader))
            except asyncio.IncompleteReadError:
                break

            result = x**2

            await P2P.send_raw_data(MSGPackSerializer.dumps(result), writer)


async def validate_square_stream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    with closing(writer):
        for _ in range(10):
            x = np.random.randint(100)

            await P2P.send_raw_data(MSGPackSerializer.dumps(x), writer)
            result = MSGPackSerializer.loads(await P2P.receive_raw_data(reader))

            assert result == x**2


@pytest.mark.asyncio
async def test_call_peer_single_process():
    server = await P2P.create()

    handler_name = "square"
    await server.add_binary_stream_handler(handler_name, handle_square_stream)

    client = await P2P.create(initial_peers=await server.get_visible_maddrs())

    await client.wait_for_at_least_n_peers(1)

    _, reader, writer = await client.call_binary_stream_handler(server.peer_id, handler_name)
    await validate_square_stream(reader, writer)

    await server.shutdown()

    await client.shutdown()


async def run_server(handler_name, server_side, response_received):
    server = await P2P.create()

    await server.add_binary_stream_handler(handler_name, handle_square_stream)

    server_side.send(server.peer_id)
    server_side.send(await server.get_visible_maddrs())
    while response_received.value == 0:
        await asyncio.sleep(0.5)

    await server.shutdown()


def server_target(handler_name, server_side, response_received):
    asyncio.run(run_server(handler_name, server_side, response_received))


@pytest.mark.asyncio
async def test_call_peer_different_processes():
    handler_name = "square"

    server_side, client_side = mp.Pipe()
    response_received = mp.Value(np.ctypeslib.as_ctypes_type(np.int32))
    response_received.value = 0

    proc = mp.Process(target=server_target, args=(handler_name, server_side, response_received))
    proc.start()

    peer_id = client_side.recv()
    peer_maddrs = client_side.recv()

    client = await P2P.create(initial_peers=peer_maddrs)

    await client.wait_for_at_least_n_peers(1)

    _, reader, writer = await client.call_binary_stream_handler(peer_id, handler_name)
    await validate_square_stream(reader, writer)

    response_received.value = 1

    await client.shutdown()

    proc.join()
    assert proc.exitcode == 0


@pytest.mark.asyncio
async def test_error_closes_connection():
    async def handle_raising_error(_, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        with closing(writer):
            command = await P2P.receive_raw_data(reader)
            if command == b"raise_error":
                raise Exception("The handler has failed")
            else:
                await P2P.send_raw_data(b"okay", writer)

    server = await P2P.create()

    handler_name = "handler"
    await server.add_binary_stream_handler(handler_name, handle_raising_error)

    client = await P2P.create(initial_peers=await server.get_visible_maddrs())

    await client.wait_for_at_least_n_peers(1)

    _, reader, writer = await client.call_binary_stream_handler(server.peer_id, handler_name)
    with closing(writer):
        await P2P.send_raw_data(b"raise_error", writer)
        with pytest.raises(asyncio.IncompleteReadError):  # Means that the connection is closed
            await P2P.receive_raw_data(reader)

    # Despite the handler raised an exception, the server did not crash and ready for next requests

    _, reader, writer = await client.call_binary_stream_handler(server.peer_id, handler_name)
    with closing(writer):
        await P2P.send_raw_data(b"behave_normally", writer)
        assert await P2P.receive_raw_data(reader) == b"okay"

    await server.shutdown()

    await client.shutdown()


@pytest.mark.asyncio
async def test_handlers_on_different_replicas():
    async def handler(_, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, key: str) -> None:
        with closing(writer):
            await P2P.send_raw_data(key, writer)

    server_primary = await P2P.create()
    server_id = server_primary.peer_id
    await server_primary.add_binary_stream_handler("handle_primary", partial(handler, key=b"primary"))

    server_replica1 = await replicate_if_needed(server_primary, True)
    await server_replica1.add_binary_stream_handler("handle1", partial(handler, key=b"replica1"))

    server_replica2 = await replicate_if_needed(server_primary, True)
    await server_replica2.add_binary_stream_handler("handle2", partial(handler, key=b"replica2"))

    client = await P2P.create(initial_peers=await server_primary.get_visible_maddrs())
    await client.wait_for_at_least_n_peers(1)

    for name, expected_key in [("handle_primary", b"primary"), ("handle1", b"replica1"), ("handle2", b"replica2")]:
        _, reader, writer = await client.call_binary_stream_handler(server_id, name)
        with closing(writer):
            assert await P2P.receive_raw_data(reader) == expected_key

    await server_replica1.shutdown()
    await server_replica2.shutdown()

    # Primary does not handle replicas protocols after their shutdown
    # (native backend fails fast at multiselect negotiation, unlike the
    # daemon which opened the stream and then reset it)

    for name in ["handle1", "handle2"]:
        with pytest.raises((P2PDaemonError, ConnectionError, asyncio.IncompleteReadError)):
            _, reader, writer = await client.call_binary_stream_handler(server_id, name)
            with closing(writer):
                await P2P.receive_raw_data(reader)

    await server_primary.shutdown()
    await client.shutdown()
