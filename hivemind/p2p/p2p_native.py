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
