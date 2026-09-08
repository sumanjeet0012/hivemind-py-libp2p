"""
Native py-libp2p transport for Hivemind (no Go daemon, pure Python).

Runs a libp2p ``BasicHost`` on a dedicated trio thread and exposes an
asyncio-compatible interface used by :class:`hivemind.p2p.P2P`.

Bridge primitives:
- unary calls: ``asyncio.to_thread(trio.from_thread.run, afn, trio_token=...)``
- byte streams: full-duplex pipe (trio pumps <-> asyncio StreamReader/Writer)
  with ``read_exactly`` loops (``INetStream.read`` may short-read).
"""

import asyncio
import queue as std_queue
import secrets
import sys
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

# Hivemind's crypto.proto and py-libp2p's crypto.proto share package
# `crypto.pb` with identical RSA/Ed25519/Secp256k1/ECDSA wire schemas
# (py-libp2p additionally defines ECC_P256/X25519). Importing both generated
# modules would double-register `crypto.pb.KeyType` in the global descriptor
# pool, so the native backend canonically uses py-libp2p's module under
# hivemind's import path. This must run before any `hivemind.proto.crypto_pb2`
# import (p2p_daemon imports this module first for exactly this reason).
import libp2p.crypto.pb.crypto_pb2 as _libp2p_crypto_pb2

sys.modules.setdefault("hivemind.proto.crypto_pb2", _libp2p_crypto_pb2)

import trio
from libp2p import generate_new_ed25519_identity, new_host
from libp2p.crypto.keys import KeyPair
from multiaddr import Multiaddr as LibMultiaddr

from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID
from hivemind.p2p.p2p_daemon_bindings.utils import P2PDaemonError
from hivemind.utils.logging import get_logger
from hivemind.utils.multiaddr import Multiaddr as HMultiaddr

logger = get_logger(__name__)

CHUNK_SIZE = 2**16
# Poll interval for the asyncio->trio byte queue. Bounds worker-thread
# abandonment on session teardown (see _pump_queue_to_libp2p).
_QUEUE_POLL_INTERVAL = 0.05
# Max buffered inbound bytes per stream. Must comfortably fit the largest
# single DHT/averaging message (daemon default max message is 4MB).
_READER_LIMIT = 2**26


def hivemind_id_to_libp2p(peer_id: PeerID):
    from libp2p.peer.id import ID

    return ID(peer_id.to_bytes())


def libp2p_id_to_hivemind(peer_id) -> PeerID:
    return PeerID(peer_id.to_bytes())


def hivemind_maddr_to_libp2p(addr) -> LibMultiaddr:
    return LibMultiaddr(str(addr))


def libp2p_maddr_to_hivemind(addr) -> HMultiaddr:
    return HMultiaddr(str(addr))


class _AsyncioWriter:
    """Minimal asyncio.StreamWriter-compatible endpoint backed by a thread-safe queue.

    ``close()`` enqueues an end-of-write sentinel so already-queued bytes are
    still flushed (half-close). It never cancels pumps outright; session
    teardown is driven by stream EOF on both directions.
    """

    def __init__(self, out_queue: "std_queue.Queue[Optional[bytes]]"):
        self._queue = out_queue
        self._closed = False

    def write(self, data: bytes) -> None:
        if self._closed:
            raise ConnectionResetError("Writer is closed")
        self._queue.put(bytes(data))

    async def drain(self) -> None:
        return None

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return default

    def is_closing(self) -> bool:
        return self._closed

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(None)

    async def wait_closed(self) -> None:
        return None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best effort on GC
            pass


@dataclass
class _StreamSession:
    loop: Any = None
    done: "threading.Event | None" = None


class TrioGateway:
    """Owns one trio event loop + one libp2p host shared by N asyncio facades."""

    def __init__(
        self,
        key_pair: Optional[KeyPair],
        listen_maddrs: List[str],
        *,
        startup_timeout: float = 15.0,
        security: str = "default",
        connection_config=None,
        announce_addrs: Optional[List[str]] = None,
        reader_limit: int = _READER_LIMIT,
    ):
        self._key_pair = key_pair
        self._listen_maddrs = listen_maddrs
        self._startup_timeout = startup_timeout
        self._security = security
        self._connection_config = connection_config
        self._announce_addrs = announce_addrs
        self._reader_limit = reader_limit
        self._ready = threading.Event()
        self._init_error: Optional[BaseException] = None
        self._trio_token = None
        self._host = None
        self._cancel_scope = None
        self._handlers: Dict[str, Callable] = {}
        self._sessions: Dict[str, _StreamSession] = {}
        # Last-known dialable addresses per peer (base58 -> multiaddr strings).
        # The peerstore may drop addresses on disconnect; this cache keeps
        # redial working for previously seen peers (e.g. DHT routing entries).
        self._known_addrs: Dict[str, List[str]] = {}
        self._rendezvous_service = None
        self._relay_protocol = None
        self._relay_transport = None
        self._refcount = 0
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._spawn_send = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="hivemind-trio-gateway")
        self._thread.start()
        if not self._ready.wait(timeout=self._startup_timeout):
            raise P2PDaemonError(
                f"Native py-libp2p host failed to start within {self._startup_timeout} seconds"
            )
        if self._init_error is not None:
            raise self._init_error

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # -- trio thread ------------------------------------------------------
    def _run(self) -> None:
        try:
            trio.run(self._main)
        except Exception as e:  # noqa: BLE001 - surface init failures to creator
            self._init_error = e
            self._ready.set()

    async def _main(self) -> None:
        self._trio_token = trio.lowlevel.current_trio_token()
        # Pumps briefly borrow worker threads for queue polling; allow headroom
        # for many concurrent streams (default limiter has only 40 tokens).
        try:
            trio.to_thread.current_default_thread_limiter().total_tokens = 256
        except Exception:  # noqa: BLE001 - best effort
            pass
        key_pair = self._key_pair or generate_new_ed25519_identity()
        sec_opt = None
        if self._security == "noise-only":
            from libp2p.crypto.x25519 import create_new_key_pair as create_new_x25519_key_pair
            from libp2p.security.noise.transport import PROTOCOL_ID as NOISE_PROTOCOL_ID
            from libp2p.security.noise.transport import Transport as NoiseTransport

            noise_kp = create_new_x25519_key_pair()
            sec_opt = {
                NOISE_PROTOCOL_ID: NoiseTransport(key_pair, noise_privkey=noise_kp.private_key),
            }
        self._host = new_host(
            key_pair=key_pair,
            sec_opt=sec_opt,
            connection_config=self._connection_config,
            announce_addrs=[LibMultiaddr(a) for a in self._announce_addrs] if self._announce_addrs else None,
        )
        addrs = [LibMultiaddr(a) for a in self._listen_maddrs]
        send, recv = trio.open_memory_channel(1024)
        self._spawn_send = send
        with trio.CancelScope() as scope:
            self._cancel_scope = scope
            async with self._host.run(listen_addrs=addrs):
                self._ready.set()
                async with trio.open_nursery() as nursery:
                    nursery.start_soon(self._supervise, recv)
                    await trio.sleep_forever()

    async def _supervise(self, recv) -> None:
        async with trio.open_nursery() as nursery:
            async for fn in recv:
                nursery.start_soon(fn)

    async def spawn(self, fn: Callable[[], Awaitable]) -> None:
        """Schedule a trio coroutine from asyncio code (fire-and-forget)."""

        async def _send() -> None:
            await self._spawn_send.send(fn)

        await asyncio.to_thread(trio.from_thread.run, _send, trio_token=self._trio_token)

    # -- unary bridge -----------------------------------------------------
    async def run(self, afn: Callable[..., Awaitable], *args, timeout: Optional[float] = None) -> Any:
        async def _wrapped():
            if timeout is not None:
                with trio.fail_after(timeout):
                    return await afn(*args)
            return await afn(*args)

        return await asyncio.to_thread(trio.from_thread.run, _wrapped, trio_token=self._trio_token)

    def run_sync(self, afn: Callable[..., Awaitable], *args) -> Any:
        return trio.from_thread.run(afn, *args, trio_token=self._trio_token)

    # -- host accessors (executed in trio) --------------------------------
    async def _get_id(self):
        return self._host.get_id()

    async def _get_addrs(self):
        return list(self._host.get_addrs())

    async def _connect(self, peer_info) -> None:
        await self._host.connect(peer_info)

    async def _new_stream(self, peer_id, protocols):
        return await self._host.new_stream(peer_id, protocols)

    async def _connected_peers(self):
        return list(self._host.get_connected_peers())

    async def _peer_addrs(self, peer_id):
        """All currently dialable addresses: live peerstore entries + dial cache."""
        try:
            live = [str(a) for a in self._host.get_peerstore().addrs(peer_id)]
        except Exception:  # noqa: BLE001 - peer may have no addrs
            live = []
        return [LibMultiaddr(a) for a in self._remember_addrs(peer_id, live)]

    def _remember_addrs(self, peer_id, addrs: List[str]) -> List[str]:
        """Merge newly observed addresses into the dial cache; return all known."""
        key = peer_id.to_base58() if hasattr(peer_id, "to_base58") else str(peer_id)
        known = self._known_addrs.setdefault(key, [])
        for addr in addrs:
            if addr not in known:
                known.append(addr)
        return list(known)

    async def _note_addrs(self, peer_id, addrs: List[str]) -> List[str]:
        return self._remember_addrs(peer_id, [str(a) for a in addrs])

    async def _disconnect(self, peer_id) -> None:
        try:
            await self._host.disconnect(peer_id)
        except Exception:  # noqa: BLE001 - peer may already be gone
            pass

    async def _set_handler(self, proto_id: str, handler) -> None:
        self._handlers[proto_id] = handler
        self._host.set_stream_handler(proto_id, handler)

    async def _remove_handler(self, proto_id: str) -> None:
        self._handlers.pop(proto_id, None)
        try:
            self._host.remove_stream_handler(proto_id)
        except Exception:  # noqa: BLE001 - best effort across versions
            pass

    async def _close_stream(self, stream) -> None:
        try:
            await stream.close()
        except Exception:  # noqa: BLE001 - already closed
            pass

    # -- rendezvous (trio side; asyncio facades go through P2P methods) -------
    async def _rendezvous_service_start(self) -> None:
        """Host a rendezvous point on this host (idempotent)."""
        if self._rendezvous_service is None:
            from libp2p.discovery.rendezvous.service import RendezvousService

            self._rendezvous_service = RendezvousService(self._host)

    async def _rendezvous_register(self, rendezvous_lib_id, namespace: str, ttl: int) -> float:
        from libp2p.discovery.rendezvous.client import RendezvousClient

        client = RendezvousClient(self._host, rendezvous_lib_id)
        try:
            return await client.register(namespace, ttl)
        finally:
            await client.close()

    async def _rendezvous_unregister(self, rendezvous_lib_id, namespace: str) -> None:
        from libp2p.discovery.rendezvous.client import RendezvousClient

        client = RendezvousClient(self._host, rendezvous_lib_id)
        try:
            await client.unregister(namespace)
        finally:
            await client.close()

    async def _rendezvous_discover(
        self, rendezvous_lib_id, namespace: str, limit: int
    ) -> List[Tuple[str, List[str]]]:
        """Discover peers; returns plain [(peer_id_b58, [addr_str])] + feeds dial cache."""
        from libp2p.discovery.rendezvous.client import RendezvousClient

        client = RendezvousClient(self._host, rendezvous_lib_id)
        try:
            peers, _cookie = await client.discover(namespace, limit=limit)
        finally:
            await client.close()
        out = []
        for peer in peers:
            addr_strs = [str(a) for a in peer.addrs]
            self._remember_addrs(peer.peer_id, addr_strs)
            out.append((peer.peer_id.to_base58(), addr_strs))
        return out

    # -- duplex byte pipe ---------------------------------------------------
    # Teardown model (drain-safe):
    # - writer.close() only enqueues an end-of-write sentinel; queued bytes
    #   are always flushed before close_write.
    # - pump_in hitting EOF cancels the nursery scope, tearing the session
    #   down and unblocking the asyncio side via feed_eof + session.done.
    def open_pipe(self, loop=None) -> Tuple[asyncio.StreamReader, _AsyncioWriter, "std_queue.Queue[Optional[bytes]]", str]:
        """Create an asyncio (reader, writer) pair bridged to trio via queues."""
        loop = loop or self._loop or asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=self._reader_limit, loop=loop)
        out_queue: "std_queue.Queue[Optional[bytes]]" = std_queue.Queue()
        session_id = secrets.token_hex(8)
        self._sessions[session_id] = _StreamSession(loop=loop, done=threading.Event())
        return reader, _AsyncioWriter(out_queue), out_queue, session_id

    def feed_reader(self, reader: asyncio.StreamReader, data: Optional[bytes], loop=None) -> None:
        loop = loop or self._loop
        if loop is None or loop.is_closed():
            return
        try:
            if data is None:
                loop.call_soon_threadsafe(reader.feed_eof)
            else:
                loop.call_soon_threadsafe(reader.feed_data, data)
        except RuntimeError:  # noqa: BLE001 - loop closed mid-shutdown
            pass

    async def _pump_libp2p_to_reader(self, stream, reader: asyncio.StreamReader,
                                     session_id: str, nursery_scope, loop=None) -> None:
        try:
            while True:
                try:
                    chunk = await stream.read(CHUNK_SIZE)
                except Exception:  # noqa: BLE001 - EOF/reset
                    break
                if not chunk:
                    break
                self.feed_reader(reader, bytes(chunk), loop)
        except Exception:  # noqa: BLE001 - surface unexpected pump failures
            logger.debug("libp2p->reader pump failed:", exc_info=True)
        finally:
            self.feed_reader(reader, None, loop)
            # Remote side ended: tear the session down so writers/consumers unblock.
            nursery_scope.cancel()

    async def _pump_queue_to_libp2p(
        self, out_queue: "std_queue.Queue[Optional[bytes]]", stream, session_id: str, nursery_scope
    ) -> None:
        # NOTE: poll with a short timeout instead of blocking get(): on session
        # teardown trio abandons the worker thread running a blocking call, so
        # a bare get() would leak one thread per closed stream. Polling bounds
        # the abandonment to POLL_INTERVAL.
        try:
            while True:
                try:
                    chunk = await trio.to_thread.run_sync(out_queue.get, True, _QUEUE_POLL_INTERVAL)
                except std_queue.Empty:
                    continue
                if chunk is None:
                    try:
                        await stream.close_write()
                    except Exception:  # noqa: BLE001
                        pass
                    break
                await stream.write(chunk)
        except Exception:  # noqa: BLE001 - stream gone
            logger.debug("reader->libp2p pump failed:", exc_info=True)

    async def _serve_stream(self, stream, reader: asyncio.StreamReader,
                            out_queue: "std_queue.Queue[Optional[bytes]]",
                            session_id: str, on_done: Optional[Callable[[], None]] = None) -> None:
        """Run duplex pumps for one libp2p stream until the session tears down."""
        session = self._sessions.get(session_id)
        loop = session.loop if session is not None else None
        try:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(self._pump_libp2p_to_reader, stream, reader, session_id,
                                   nursery.cancel_scope, loop)
                nursery.start_soon(self._pump_queue_to_libp2p, out_queue, stream, session_id, nursery.cancel_scope)
        finally:
            self.feed_reader(reader, None, loop)
            session = self._sessions.pop(session_id, None)
            if session is not None and session.done is not None:
                session.done.set()
            await self._close_stream(stream)
            if on_done is not None:
                on_done()

    # -- circuit relay v2 (trio side) ----------------------------------------
    async def _relay_setup(self, allow_hop: bool) -> None:
        """Enable relay roles on this host (idempotent; upgrade-only)."""
        from libp2p.relay.circuit_v2.config import RelayConfig, RelayRole
        from libp2p.relay.circuit_v2.protocol import (
            PROTOCOL_ID as RELAY_PROTOCOL_ID,
        )
        from libp2p.relay.circuit_v2.protocol import (
            STOP_PROTOCOL_ID as RELAY_STOP_ID,
        )
        from libp2p.relay.circuit_v2.protocol import CircuitV2Protocol
        from libp2p.relay.circuit_v2.transport import CircuitV2Transport

        if self._relay_transport is not None:
            if allow_hop and not self._relay_protocol.allow_hop:
                self._relay_protocol.allow_hop = True
            return
        roles = (RelayRole.HOP | RelayRole.STOP | RelayRole.CLIENT) if allow_hop \
            else (RelayRole.STOP | RelayRole.CLIENT)
        config = RelayConfig(roles=roles)
        protocol = CircuitV2Protocol(self._host, None, allow_hop=allow_hop)
        self._relay_transport = CircuitV2Transport(self._host, protocol, config)
        self._host.set_stream_handler(RELAY_PROTOCOL_ID, protocol._handle_hop_stream)
        self._host.set_stream_handler(RELAY_STOP_ID, protocol._handle_stop_stream)
        self._relay_protocol = protocol

    async def _relay_has_reservation(self, peer_id) -> bool:
        if self._relay_protocol is None:
            return False
        try:
            return bool(self._relay_protocol.resource_manager.has_reservation(peer_id))
        except Exception:  # noqa: BLE001
            return False

    async def _relay_reserve(self, relay_lib_id) -> str:
        """Reserve a relay slot; returns our relay address via this relay."""
        from libp2p.peer.peerstore import env_to_send_in_RPC
        from libp2p.relay.circuit_v2.pb.circuit_pb2 import HopMessage
        from libp2p.relay.circuit_v2.protocol import PROTOCOL_ID as RELAY_PROTOCOL_ID

        if self._relay_transport is None:
            await self._relay_setup(allow_hop=False)
        stream = await self._host.new_stream(relay_lib_id, [RELAY_PROTOCOL_ID])
        try:
            envelope_bytes, _ = env_to_send_in_RPC(self._host)
            msg = HopMessage(
                type=HopMessage.RESERVE,
                peer=self._host.get_id().to_bytes(),
                senderRecord=envelope_bytes,
            )
            await stream.write(msg.SerializeToString())
            with trio.fail_after(15):
                resp_bytes = await stream.read(4096)
            resp = HopMessage()
            resp.ParseFromString(resp_bytes)
            if resp.type != HopMessage.STATUS:
                raise RuntimeError(f"Unexpected relay response type: {resp.type}")
        finally:
            await stream.close()
        relay_addrs = await self._peer_addrs(relay_lib_id)
        if not relay_addrs:
            raise RuntimeError("No known address for relay peer")
        relay_b58 = relay_lib_id.to_base58()
        return f"{relay_addrs[0]}/p2p/{relay_b58}/p2p-circuit/p2p/{self._host.get_id().to_base58()}"

    async def _relay_dial(self, relay_addr_str: str) -> None:
        """Dial a peer through a relay; registers the relayed route for later ID-dials."""
        from libp2p.peer.id import ID as LibID

        if self._relay_transport is None:
            await self._relay_setup(allow_hop=False)
        addr = LibMultiaddr(relay_addr_str)
        await self._relay_transport.dial(addr)
        dest_b58 = relay_addr_str.rsplit("/p2p/", 1)[-1]
        dest_id = LibID.from_string(dest_b58) if hasattr(LibID, "from_string") else LibID(dest_b58)
        try:
            self._host.get_peerstore().add_addr(dest_id, addr, 3600)
        except Exception:  # noqa: BLE001 - best effort
            pass
        self._remember_addrs(dest_id, [relay_addr_str])

    # -- autonat reachability (trio side) ---------------------------------------
    async def _autonat_serve(self) -> None:
        """Answer AutoNAT dial-back requests on this host (idempotent)."""
        if getattr(self, "_autonat_service", None) is not None:
            return
        from libp2p.host.autonat.autonat import AUTONAT_PROTOCOL_ID, AutoNATService

        service = AutoNATService(self._host)
        self._host.set_stream_handler(AUTONAT_PROTOCOL_ID, service.handle_stream)
        self._autonat_service = service

    async def _autonat_status(self) -> str:
        service = getattr(self, "_autonat_service", None)
        if service is None:
            return "unknown"
        try:
            status = service.get_status()
        except Exception:  # noqa: BLE001
            return "unknown"
        return {0: "unknown", 1: "public", 2: "private"}.get(int(status), "unknown")

    async def _autonat_check(self, server_lib_id, addr_strs: List[str]) -> bool:
        """Ask a server to dial us back at ``addr_strs``; True if it reached us."""
        from libp2p.host.autonat.autonat import AUTONAT_PROTOCOL_ID
        from libp2p.host.autonat.pb.autonat_pb2 import DialResponse, Message, Status, Type

        request = Message(type=Type.DIAL)
        dial = request.dial
        self_id = self._host.get_id()
        entry = dial.peers.add()
        entry.id = self_id.to_bytes()
        entry.addrs.extend([a.encode() for a in addr_strs])
        stream = await self._host.new_stream(server_lib_id, [AUTONAT_PROTOCOL_ID])
        try:
            await stream.write(request.SerializeToString())
            with trio.fail_after(20):
                response_bytes = await stream.read(65536)
            response = Message()
            response.ParseFromString(response_bytes)
            if response.type != Type.DIAL_RESPONSE:
                return False
            dial_resp: DialResponse = response.dial_response
            if dial_resp.status != Status.OK:
                return False
            return any(getattr(p, "success", False) for p in dial_resp.peers)
        finally:
            await stream.close()

    # -- dcutr hole punching (trio side) ----------------------------------------
    async def _dcutr_start(self) -> None:
        """Run the DCUtR responder on this host (idempotent)."""
        if getattr(self, "_dcutr", None) is not None:
            return
        from libp2p.relay.circuit_v2.dcutr import DCUtRProtocol
        from libp2p.relay.circuit_v2.dcutr import PROTOCOL_ID as DCUTR_PROTOCOL_ID

        dcutr = DCUtRProtocol(self._host)
        # Register only the responder; the full Service.run() needs a service
        # manager, unnecessary for answering hole-punch streams in tests.
        self._host.set_stream_handler(DCUTR_PROTOCOL_ID, dcutr._handle_dcutr_stream)
        self._dcutr = dcutr

    async def _dcutr_punch(self, peer_lib_id) -> bool:
        from libp2p.relay.circuit_v2.dcutr import DCUtRProtocol

        dcutr = getattr(self, "_dcutr", None)
        if dcutr is None:
            dcutr = DCUtRProtocol(self._host)
            self._dcutr = dcutr
        try:
            with trio.fail_after(30):
                return bool(await dcutr.initiate_hole_punch(peer_lib_id))
        except Exception:  # noqa: BLE001 - punch failed
            logger.debug("DCUtR hole punch failed:", exc_info=True)
            return False

    # -- lifecycle ----------------------------------------------------------
    def retain(self) -> None:
        with self._lock:
            self._refcount += 1

    def release(self) -> None:
        with self._lock:
            self._refcount -= 1
            remaining = self._refcount
        if remaining <= 0:
            self.stop()

    def stop(self) -> None:
        # Non-blocking by design: fire the cancel into trio and return.
        # A blocking roundtrip (trio.from_thread.run) is unsafe here because
        # stop() runs on refcount release, including from P2P.__del__ during
        # interpreter shutdown when trio may no longer answer.
        try:
            token, scope = self._trio_token, self._cancel_scope
            if token is not None and scope is not None:
                token.run_sync_soon(scope.cancel)
        except Exception:  # noqa: BLE001 - already stopped
            pass
