import asyncio
import os
import secrets
import threading
import warnings
from collections.abc import AsyncIterable as AsyncIterableABC
from contextlib import closing, suppress
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, List, Optional, Sequence, Tuple, Type, TypeVar, Union

from google.protobuf.message import Message

from hivemind.p2p.p2p_native import (  # noqa: E402 - first: installs crypto_pb2 alias
    TrioGateway,
    _READER_LIMIT,
    hivemind_id_to_libp2p,
    hivemind_maddr_to_libp2p,
    libp2p_id_to_hivemind,
    libp2p_maddr_to_hivemind,
)
from hivemind.p2p.p2p_daemon_bindings.control import DEFAULT_MAX_MSG_SIZE, P2PDaemonError, P2PHandlerError
from hivemind.p2p.p2p_daemon_bindings.datastructures import PeerID, PeerInfo, StreamInfo
from hivemind.proto import crypto_pb2
from hivemind.proto.p2pd_pb2 import RPCError
from hivemind.utils.asyncio import as_aiter, asingle
from hivemind.utils.crypto import RSAPrivateKey
from hivemind.utils.logging import get_logger
from hivemind.utils.multiaddr import Multiaddr

logger = get_logger(__name__)


@dataclass(frozen=True)
class P2PContext:
    handle_name: str
    local_id: PeerID
    remote_id: PeerID = None


# Maps daemon listen tokens (opaque in-process identifiers) to shared gateways.
# Replaces the old "connect to an already running p2pd" mechanism.
_SHARED_GATEWAYS: dict = {}

# Params accepted for backward compatibility but not yet wired to py-libp2p
# (Rendezvous / NAT traversal land in Phases 3-4). Each is warned about once
# per process when explicitly set to a non-default value.
_NOOP_WARNED: set = set()


def _warn_noop_once(name: str, detail: str) -> None:
    if name not in _NOOP_WARNED:
        _NOOP_WARNED.add(name)
        warnings.warn(
            f"hivemind.P2P parameter `{name}` is accepted for backward compatibility "
            f"but has no effect in the native py-libp2p backend yet ({detail}).",
            UserWarning,
            stacklevel=3,
        )


def _warn_balanced_once() -> None:
    _warn_noop_once("balanced", "single in-process host needs no cross-handler balancing")


def _proto(name: str) -> str:
    """Single wire namespace for unary, streaming and raw binary handlers.

    Mirrors the Go daemon, where ``add_unary_handler`` and ``stream_handler``
    share one name registry.
    """
    return name if name.startswith("/") else f"/hivemind/{name}"


class P2P:
    """
    Native py-libp2p peer (pure Python, no Go daemon).

    Runs an in-process libp2p host on a dedicated trio thread (see
    :mod:`hivemind.p2p.p2p_native`) while exposing the same asyncio API that
    higher Hivemind layers (DHT, averaging, MoE) already use.
    """

    HEADER_LEN = 8
    BYTEORDER = "big"
    MESSAGE_MARKER = b"\x00"
    ERROR_MARKER = b"\x01"
    END_OF_STREAM = RPCError()

    DHT_MODE_MAPPING = {
        "auto": {"dht": 1},
        "server": {"dhtServer": 1},
        "client": {"dhtClient": 1},
    }
    FORCE_REACHABILITY_MAPPING = {
        "public": {"forceReachabilityPublic": 1},
        "private": {"forceReachabilityPrivate": 1},
    }
    _UNIX_SOCKET_PREFIX = "/unix/tmp/hivemind-"

    def __init__(self):
        self.peer_id = None
        self._gateway: Optional[TrioGateway] = None
        self._daemon_listen_maddr = None
        self._visible_maddrs: List[Multiaddr] = []
        self._alive = False
        self._stream_handlers: dict = {}

    @classmethod
    async def create(
        cls,
        initial_peers: Optional[Sequence[Union[Multiaddr, str]]] = None,
        *,
        announce_maddrs: Optional[Sequence[Union[Multiaddr, str]]] = None,
        auto_nat: bool = True,
        conn_manager: bool = True,
        dht_mode: str = "server",
        force_reachability: Optional[str] = None,
        host_maddrs: Optional[Sequence[Union[Multiaddr, str]]] = ("/ip4/127.0.0.1/tcp/0",),
        identity_path: Optional[str] = None,
        idle_timeout: float = 30,
        nat_port_map: bool = True,
        relay_hop_limit: int = 0,
        startup_timeout: float = 15,
        tls: bool = True,
        use_auto_relay: bool = False,
        use_ipfs: bool = False,
        use_relay: bool = True,
        persistent_conn_max_msg_size: int = DEFAULT_MAX_MSG_SIZE,
        quic: Optional[bool] = None,
        use_relay_hop: Optional[bool] = None,
        use_relay_discovery: Optional[bool] = None,
        check_if_identity_free: bool = True,
        no_listen: bool = False,
        trusted_relays: Optional[Sequence[Union[Multiaddr, str]]] = None,
    ) -> "P2P":
        """
        Start an in-process native py-libp2p host and connect to the swarm.

        Parameter mapping onto py-libp2p: ``host_maddrs`` → listen addrs (TCP/QUIC
        auto-detected), ``announce_maddrs`` → announced addrs, ``identity_path`` →
        RSA keypair, ``tls`` → Noise+TLS (default) or Noise-only, ``conn_manager`` →
        default connection manager, ``startup_timeout`` → host ready timeout,
        ``persistent_conn_max_msg_size`` → inbound stream buffer cap (when set),
        ``no_listen`` → no listen addrs, ``persistent_conn_max_msg_size`` → inbound
        stream buffer cap when explicitly set. The remaining transport/discovery flags
        (``auto_nat``, ``use_relay``, ``dht_mode``, …) are accepted for signature
        compatibility and warn once when set to non-default values; they are
        wired up in later phases (Rendezvous / NAT traversal).
        """
        assert not (initial_peers and use_ipfs), (
            "User-defined initial_peers and use_ipfs=True are incompatible, please choose one option"
        )
        if use_ipfs:
            raise P2PDaemonError("use_ipfs is not supported by the native py-libp2p backend")

        if not all(arg is None for arg in [quic, use_relay_hop, use_relay_discovery]):
            warnings.warn(
                "Parameters `quic`, `use_relay_hop`, and `use_relay_discovery` of hivemind.P2P "
                "have no effect since libp2p 0.17.0 and will be removed in hivemind 1.2.0+",
                DeprecationWarning,
                stacklevel=2,
            )
        if dht_mode not in cls.DHT_MODE_MAPPING:
            raise ValueError(f"Unknown dht_mode {dht_mode!r}, expected one of {sorted(cls.DHT_MODE_MAPPING)}")
        if force_reachability is not None and force_reachability not in cls.FORCE_REACHABILITY_MAPPING:
            raise ValueError(f"Unknown force_reachability {force_reachability!r}")

        # Honest no-ops: warn once per process when explicitly set off-default.
        if dht_mode != "server":
            _warn_noop_once("dht_mode", "Kademlia mode selection needs Phase-3 discovery")
        if force_reachability is not None:
            _warn_noop_once("force_reachability", "reachability forcing needs Phase-4 AutoNAT")
        if not auto_nat:
            _warn_noop_once("auto_nat", "AutoNAT service needs Phase-4 wiring")
        if not use_relay:
            _warn_noop_once("use_relay", "Circuit Relay v2 needs Phase-4 wiring")
        if use_auto_relay:
            _warn_noop_once("use_auto_relay", "auto-relay needs Phase-4 wiring")
        if not nat_port_map:
            _warn_noop_once("nat_port_map", "port mapping needs Phase-4 wiring")
        if relay_hop_limit:
            _warn_noop_once("relay_hop_limit", "relay limits need Phase-4 wiring")
        if trusted_relays is not None:
            _warn_noop_once("trusted_relays", "relay selection needs Phase-4 wiring")
        if not conn_manager:
            _warn_noop_once(
                "conn_manager", "the swarm always runs its default connection manager"
            )
        if idle_timeout != 30:
            _warn_noop_once("idle_timeout", "no persistent-conn sweeps in the native backend")

        self = cls()
        if announce_maddrs is not None:
            for addr in announce_maddrs:
                addr = Multiaddr(addr)
                if ("tcp" in addr and addr["tcp"] == "0") or ("udp" in addr and addr["udp"] == "0"):
                    raise ValueError("Please specify an explicit port in announce_maddrs: port 0 is not supported")

        key_pair = None
        if identity_path is not None:
            if os.path.isfile(identity_path):
                if check_if_identity_free and initial_peers:
                    logger.info(f"Checking that identity from `{identity_path}` is not used by other peers")
                    if await cls.is_identity_taken(identity_path, initial_peers=initial_peers):
                        raise P2PDaemonError(f"Identity from `{identity_path}` is already taken by another peer")
            else:
                logger.info(f"Generating new identity to be saved in `{identity_path}`")
                self.generate_identity(identity_path)
            key_pair = cls._load_keypair(identity_path)

        listen_maddrs = [] if no_listen else [str(a) for a in (host_maddrs or [])]
        announce_addrs = None
        if announce_maddrs is not None:
            announce_addrs = [str(Multiaddr(a)) for a in announce_maddrs]
        gateway = TrioGateway(
            key_pair=key_pair,
            listen_maddrs=listen_maddrs,
            startup_timeout=startup_timeout,
            security="default" if tls else "noise-only",
            connection_config=None,  # swarm default connection manager
            announce_addrs=announce_addrs,
            # The daemon capped persistent-conn frames at this size; natively it
            # bounds inbound stream buffering. Default (4MB) keeps the generous
            # 64MB native buffer; an explicit value is honored literally.
            reader_limit=(
                _READER_LIMIT if persistent_conn_max_msg_size == DEFAULT_MAX_MSG_SIZE
                else persistent_conn_max_msg_size
            ),
        )
        gateway.attach_loop(asyncio.get_running_loop())
        gateway.retain()
        self._gateway = gateway

        token_uid = secrets.token_urlsafe(8)
        self._daemon_listen_maddr = Multiaddr(cls._UNIX_SOCKET_PREFIX + f"native-{token_uid}.sock")
        _SHARED_GATEWAYS[str(self._daemon_listen_maddr)] = gateway

        lib_id = await gateway.run(gateway._get_id)
        self.peer_id = libp2p_id_to_hivemind(lib_id)

        raw = await gateway.run(gateway._get_addrs)
        self._visible_maddrs = [libp2p_maddr_to_hivemind(a) for a in raw]
        p2p_suffix = Multiaddr(f"/p2p/{self.peer_id.to_base58()}")
        self._visible_maddrs = [a.encapsulate(p2p_suffix) if "p2p" not in a else a for a in self._visible_maddrs]

        self._alive = True
        logger.debug(f"Launched native py-libp2p host with peer id = {self.peer_id}")

        if initial_peers:
            for addr in initial_peers:
                with suppress(Exception):
                    await self._connect_to_maddr(Multiaddr(addr))
        return self

    @staticmethod
    def _load_keypair(identity_path: str):
        from Crypto.PublicKey import RSA

        from libp2p.crypto.keys import KeyPair
        from libp2p.crypto.rsa import RSAPrivateKey as LibRSAPrivateKey

        with open(identity_path, "rb") as f:
            protobuf = crypto_pb2.PrivateKey.FromString(f.read())
        if protobuf.key_type != crypto_pb2.RSA:
            raise P2PDaemonError(f"Unsupported key type in `{identity_path}` (native backend supports RSA)")
        impl = RSA.import_key(protobuf.data)
        private = LibRSAPrivateKey(impl)
        return KeyPair(private_key=private, public_key=private.get_public_key())

    @classmethod
    async def is_identity_taken(
        cls,
        identity_path: str,
        *,
        initial_peers: Optional[Sequence[Union[Multiaddr, str]]] = None,
        tls: bool = True,
        use_auto_relay: bool = False,
        use_ipfs: bool = False,
        use_relay: bool = True,
    ) -> bool:
        if use_ipfs:
            raise P2PDaemonError("use_ipfs is not supported by the native py-libp2p backend")
        if use_auto_relay:
            _warn_noop_once("use_auto_relay", "auto-relay needs Phase-4 wiring")
        if not use_relay:
            _warn_noop_once("use_relay", "Circuit Relay v2 needs Phase-4 wiring")
        with open(identity_path, "rb") as f:
            peer_id = PeerID.from_identity(f.read())

        anonymous = await cls.create(initial_peers=initial_peers, check_if_identity_free=False, tls=tls)
        try:
            for _ in range(50):
                peers = await anonymous.list_peers()
                if any(p.peer_id == peer_id for p in peers):
                    return True
                await asyncio.sleep(0.1)
            return False
        finally:
            await anonymous.shutdown()

    @staticmethod
    def generate_identity(identity_path: str) -> None:
        private_key = RSAPrivateKey()
        protobuf = crypto_pb2.PrivateKey(key_type=crypto_pb2.KeyType.RSA, data=private_key.to_bytes())

        try:
            with open(identity_path, "wb") as f:
                f.write(protobuf.SerializeToString())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The directory `{os.path.dirname(identity_path)}` for saving the identity does not exist"
            )
        os.chmod(identity_path, 0o400)

    @classmethod
    async def replicate(cls, daemon_listen_maddr: Multiaddr) -> "P2P":
        """
        Attach to an existing in-process native host.

        :param daemon_listen_maddr: token returned as ``P2P.daemon_listen_maddr``
               by another native ``P2P`` instance in this process.
        """
        gateway = _SHARED_GATEWAYS.get(str(daemon_listen_maddr))
        if gateway is None:
            raise P2PDaemonError(f"No native py-libp2p host found for {daemon_listen_maddr}")

        self = cls()
        gateway.attach_loop(asyncio.get_running_loop())
        gateway.retain()
        self._gateway = gateway
        self._daemon_listen_maddr = Multiaddr(daemon_listen_maddr)

        lib_id = await gateway.run(gateway._get_id)
        self.peer_id = libp2p_id_to_hivemind(lib_id)
        raw = await gateway.run(gateway._get_addrs)
        p2p_suffix = Multiaddr(f"/p2p/{self.peer_id.to_base58()}")
        self._visible_maddrs = []
        for a in raw:
            ha = libp2p_maddr_to_hivemind(a)
            self._visible_maddrs.append(ha.encapsulate(p2p_suffix) if "p2p" not in ha else ha)
        self._alive = True
        return self

    async def _connect_to_maddr(self, addr: Multiaddr) -> None:
        from libp2p.peer.peerinfo import PeerInfo as LibPeerInfo

        addr_str = str(addr)
        if "/p2p/" not in addr_str:
            raise P2PDaemonError(f"Cannot connect to {addr_str}: missing /p2p/ peer ID component")
        base, b58 = addr_str.rsplit("/p2p/", 1)
        lib_id = hivemind_id_to_libp2p(PeerID.from_base58(b58))
        lib_addr = hivemind_maddr_to_libp2p(Multiaddr(base))
        try:
            await self._gateway.run(self._gateway._connect, LibPeerInfo(lib_id, [lib_addr]))
            await self._gateway.run(self._gateway._note_addrs, lib_id, [str(lib_addr)])
        except Exception as e:  # noqa: BLE001 - best effort bootstrap
            logger.warning(f"Failed to connect to bootstrap peer {addr_str}: {e}")

    async def _ensure_connected(self, peer_id: PeerID) -> None:
        peers = await self.list_peers()
        if any(p.peer_id == peer_id for p in peers):
            return
        # Fall back to peerstore addresses (populated via identify on prior contact)
        from libp2p.peer.peerinfo import PeerInfo as LibPeerInfo

        lib_id = hivemind_id_to_libp2p(peer_id)

        async def _connect_known() -> None:
            addrs = await self._gateway.run(self._gateway._peer_addrs, lib_id)
            if not addrs:
                raise P2PDaemonError(f"Not connected to peer {peer_id.to_base58()} and no known addresses")
            await self._gateway.run(self._gateway._connect, LibPeerInfo(lib_id, addrs))

        await _connect_known()

    @property
    def daemon_listen_maddr(self) -> Multiaddr:
        return self._daemon_listen_maddr

    async def get_visible_maddrs(self, latest: bool = False) -> List[Multiaddr]:
        """
        Get multiaddrs of the current peer that should be accessible by other peers.

        :param latest: re-read the addresses from the live host instead of the
                       cached snapshot taken at startup.
        """
        if latest and self._gateway is not None:
            raw = await self._gateway.run(self._gateway._get_addrs)
            p2p_suffix = Multiaddr(f"/p2p/{self.peer_id.to_base58()}")
            self._visible_maddrs = []
            for a in raw:
                ha = libp2p_maddr_to_hivemind(a)
                self._visible_maddrs.append(ha.encapsulate(p2p_suffix) if "p2p" not in ha else ha)

        if not self._visible_maddrs:
            raise ValueError(f"No multiaddrs found for peer {self.peer_id}")
        return list(self._visible_maddrs)

    async def list_peers(self) -> List[PeerInfo]:
        gateway = self._gateway
        lib_ids = await gateway.run(gateway._connected_peers)
        peers = []
        for lib_id in lib_ids:
            raw_addrs = await gateway.run(gateway._peer_addrs, lib_id)
            peers.append(
                PeerInfo(
                    libp2p_id_to_hivemind(lib_id),
                    [libp2p_maddr_to_hivemind(a) for a in raw_addrs],
                )
            )
        return peers

    async def wait_for_at_least_n_peers(self, n_peers: int, attempts: int = 3, delay: float = 1) -> None:
        for _ in range(attempts):
            if len(await self.list_peers()) >= n_peers:
                return
            await asyncio.sleep(delay)
        raise RuntimeError("Not enough peers")

    @staticmethod
    async def send_raw_data(data: bytes, writer: asyncio.StreamWriter, *, chunk_size: int = 2**16) -> None:
        writer.write(len(data).to_bytes(P2P.HEADER_LEN, P2P.BYTEORDER))
        data = memoryview(data)
        for offset in range(0, len(data), chunk_size):
            writer.write(data[offset : offset + chunk_size])
        await writer.drain()

    @staticmethod
    async def receive_raw_data(reader: asyncio.StreamReader) -> bytes:
        header = await reader.readexactly(P2P.HEADER_LEN)
        content_length = int.from_bytes(header, P2P.BYTEORDER)
        data = await reader.readexactly(content_length)
        return data

    TInputProtobuf = TypeVar("TInputProtobuf")
    TOutputProtobuf = TypeVar("TOutputProtobuf")

    @staticmethod
    async def send_protobuf(protobuf: Union[TOutputProtobuf, RPCError], writer: asyncio.StreamWriter) -> None:
        if isinstance(protobuf, RPCError):
            writer.write(P2P.ERROR_MARKER)
        else:
            writer.write(P2P.MESSAGE_MARKER)
        await P2P.send_raw_data(protobuf.SerializeToString(), writer)

    @staticmethod
    async def receive_protobuf(
        input_protobuf_type: Type[Message], reader: asyncio.StreamReader
    ) -> Tuple[Optional[TInputProtobuf], Optional[RPCError]]:
        msg_type = await reader.readexactly(1)
        if msg_type == P2P.MESSAGE_MARKER:
            protobuf = input_protobuf_type()
            protobuf.ParseFromString(await P2P.receive_raw_data(reader))
            return protobuf, None
        elif msg_type == P2P.ERROR_MARKER:
            protobuf = RPCError()
            protobuf.ParseFromString(await P2P.receive_raw_data(reader))
            return None, protobuf
        else:
            raise TypeError("Invalid Protobuf message type")

    TInputStream = AsyncIterator[TInputProtobuf]
    TOutputStream = AsyncIterator[TOutputProtobuf]

    async def _add_protobuf_stream_handler(
        self,
        name: str,
        handler: Callable[[TInputStream, P2PContext], TOutputStream],
        input_protobuf_type: Type[Message],
        max_prefetch: int = 5,
        balanced: bool = False,
    ) -> None:
        async def _handle_stream(
            stream_info: StreamInfo, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            context = P2PContext(
                handle_name=name,
                local_id=self.peer_id,
                remote_id=stream_info.peer_id,
            )
            requests = asyncio.Queue(max_prefetch)

            async def _read_stream() -> P2P.TInputStream:
                while True:
                    request = await requests.get()
                    if request is None:
                        break
                    yield request

            async def _process_stream() -> None:
                try:
                    async for response in handler(_read_stream(), context):
                        try:
                            await P2P.send_protobuf(response, writer)
                        except Exception:
                            logger.debug("Exception while sending response:", exc_info=True)
                            break
                except Exception as e:
                    logger.warning("Handler failed with the exception:", exc_info=True)
                    with suppress(Exception):
                        await P2P.send_protobuf(RPCError(message=str(e)), writer)

            with closing(writer):
                processing_task = asyncio.create_task(_process_stream())
                try:
                    while True:
                        receive_task = asyncio.create_task(P2P.receive_protobuf(input_protobuf_type, reader))
                        await asyncio.wait({processing_task, receive_task}, return_when=asyncio.FIRST_COMPLETED)

                        if processing_task.done():
                            receive_task.cancel()
                            return

                        if receive_task.done():
                            try:
                                request, _ = await receive_task
                            except asyncio.IncompleteReadError:  # Connection is closed (the client cancelled or died)
                                return
                            await requests.put(request)  # `request` is None for the end-of-stream message
                except Exception:
                    logger.warning("Exception while receiving requests:", exc_info=True)
                finally:
                    processing_task.cancel()

        await self.add_binary_stream_handler(_proto(name), _handle_stream, balanced=balanced)

    async def _iterate_protobuf_stream_handler(
        self, peer_id: PeerID, name: str, requests: TInputStream, output_protobuf_type: Type[Message]
    ) -> TOutputStream:
        _, reader, writer = await self.call_binary_stream_handler(peer_id, name)

        async def _write_to_stream() -> None:
            try:
                async for request in requests:
                    await P2P.send_protobuf(request, writer)
                await P2P.send_protobuf(P2P.END_OF_STREAM, writer)
            except (ConnectionResetError, BrokenPipeError):
                pass

        async def _read_from_stream() -> AsyncIterator[Message]:
            with closing(writer):
                try:
                    while True:
                        try:
                            response, err = await P2P.receive_protobuf(output_protobuf_type, reader)
                        except asyncio.IncompleteReadError:
                            break

                        if err is not None:
                            raise P2PHandlerError(f"Failed to call handler `{name}` at {peer_id}: {err.message}")
                        yield response

                    await writing_task
                finally:
                    writing_task.cancel()

        writing_task = asyncio.create_task(_write_to_stream())
        return _read_from_stream()

    async def add_protobuf_handler(
        self,
        name: str,
        handler: Callable[
            [Union[TInputProtobuf, TInputStream], P2PContext], Union[Awaitable[TOutputProtobuf], TOutputStream]
        ],
        input_protobuf_type: Type[Message],
        *,
        stream_input: bool = False,
        stream_output: bool = False,
        balanced: bool = False,
    ) -> None:
        # Like the Go daemon, unary and streaming handlers share one namespace:
        # every protobuf handler is served by the streaming engine, with unary
        # calls mapped onto a single request/response exchange.
        if balanced:
            _warn_balanced_once()
        async def _stream_handler(requests: P2P.TInputStream, context: P2PContext) -> P2P.TOutputStream:
            input = requests if stream_input else await asingle(requests)
            output = handler(input, context)

            if isinstance(output, AsyncIterableABC):
                async for item in output:
                    yield item
            else:
                yield await output

        await self._add_protobuf_stream_handler(name, _stream_handler, input_protobuf_type, balanced=balanced)

    async def remove_protobuf_handler(
        self,
        name: str,
        *,
        stream_input: bool = False,
        stream_output: bool = False,
    ) -> None:
        await self.remove_binary_stream_handler(_proto(name))

    async def call_protobuf_handler(
        self,
        peer_id: PeerID,
        name: str,
        input: Union[TInputProtobuf, TInputStream],
        output_protobuf_type: Type[Message],
    ) -> Awaitable[TOutputProtobuf]:
        if not isinstance(input, AsyncIterableABC):
            return await self._call_unary_protobuf_handler(peer_id, name, input, output_protobuf_type)

        responses = await self._iterate_protobuf_stream_handler(peer_id, name, input, output_protobuf_type)
        return await asingle(responses)

    async def _call_unary_protobuf_handler(
        self,
        peer_id: PeerID,
        handle_name: str,
        input: TInputProtobuf,
        output_protobuf_type: Type[Message],
    ) -> Awaitable[TOutputProtobuf]:
        if peer_id == self.peer_id:
            raise P2PDaemonError("Cannot dial self")
        await self._ensure_connected(peer_id)
        _, reader, writer = await self.call_binary_stream_handler(peer_id, _proto(handle_name))
        with closing(writer):
            try:
                await P2P.send_protobuf(input, writer)
                # Terminate the request stream so the server's single-request
                # adapter (asingle) completes after exactly one item.
                await P2P.send_protobuf(P2P.END_OF_STREAM, writer)
                response, err = await P2P.receive_protobuf(output_protobuf_type, reader)
            except asyncio.CancelledError:
                writer.close()
                raise
        if err is not None:
            raise P2PHandlerError(f"Failed to call handler `{handle_name}` at {peer_id}: {err.message}")
        if response is None:
            raise P2PDaemonError(f"No response from handler `{handle_name}` at {peer_id}")
        return response

    async def iterate_protobuf_handler(
        self,
        peer_id: PeerID,
        name: str,
        input: Union[TInputProtobuf, TInputStream],
        output_protobuf_type: Type[Message],
    ) -> TOutputStream:
        if peer_id == self.peer_id:
            raise P2PDaemonError("Cannot dial self")
        await self._ensure_connected(peer_id)
        requests = input if isinstance(input, AsyncIterableABC) else as_aiter(input)
        return await self._iterate_protobuf_stream_handler(peer_id, name, requests, output_protobuf_type)

    async def add_binary_stream_handler(
        self, name: str, handler, balanced: bool = False
    ) -> None:
        if balanced:
            _warn_balanced_once()
        name = _proto(name)
        gateway = self._gateway
        if gateway is None:
            raise P2PDaemonError("P2P instance is shut down")
        if name in gateway._handlers or name in self._stream_handlers:
            raise P2PDaemonError(f"Handler `{name}` is already registered")
        loop = asyncio.get_running_loop()
        self._stream_handlers[name] = handler

        async def _trio_on_stream(stream) -> None:
            try:
                remote_lib_id = stream.muxed_conn.peer_id
            except Exception:  # noqa: BLE001 - defensive
                remote_lib_id = None
            try:
                remote = libp2p_id_to_hivemind(remote_lib_id) if remote_lib_id is not None else self.peer_id
            except Exception:  # noqa: BLE001 - defensive
                remote = self.peer_id
            info = StreamInfo(peer_id=remote, addr=Multiaddr("/ip4/127.0.0.1/tcp/0"), proto=name)
            reader, writer, out_queue, session_id = gateway.open_pipe(loop=loop)
            closed = threading.Event()
            setattr(reader, "_native_stream_closed", closed)  # noqa: B010 - bridge bookkeeping

            async def _consume() -> None:
                try:
                    await handler(info, reader, writer)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - user handler errors are logged
                    logger.warning("Binary stream handler failed:", exc_info=True)
                finally:
                    with suppress(Exception):
                        writer.close()

            consumer = asyncio.run_coroutine_threadsafe(_consume(), loop)
            try:
                await gateway._serve_stream(stream, reader, out_queue, session_id)
            finally:
                closed.set()
                if not consumer.done():
                    consumer.cancel()

        async def _set() -> None:
            await gateway._set_handler(name, _trio_on_stream)

        await gateway.run(_set)

    async def remove_binary_stream_handler(self, name: str) -> None:
        name = _proto(name)
        if name not in self._stream_handlers:
            raise P2PDaemonError(f"Handler `{name}` is not registered")
        del self._stream_handlers[name]
        gateway = self._gateway

        async def _remove() -> None:
            await gateway._remove_handler(name)

        await gateway.run(_remove)

    async def call_binary_stream_handler(
        self, peer_id: PeerID, handler_name: str
    ) -> Tuple[StreamInfo, asyncio.StreamReader, asyncio.StreamWriter]:
        if peer_id == self.peer_id:
            raise P2PDaemonError("Cannot dial self")
        await self._ensure_connected(peer_id)
        gateway = self._gateway
        handler_name = _proto(handler_name)
        stream = await self._open_binary_stream(peer_id, handler_name)
        reader, writer, out_queue, session_id = gateway.open_pipe()
        closed = threading.Event()
        setattr(reader, "_native_stream_closed", closed)  # noqa: B010 - bridge bookkeeping

        async def _serve() -> None:
            await gateway._serve_stream(stream, reader, out_queue, session_id)
            closed.set()

        await gateway.spawn(_serve)
        info = StreamInfo(peer_id=peer_id, addr=Multiaddr("/ip4/127.0.0.1/tcp/0"), proto=handler_name)
        return info, reader, writer

    async def _reconnect(self, peer_id: PeerID) -> None:
        """Drop stale connection state and reconnect via peerstore addresses."""
        from libp2p.peer.peerinfo import PeerInfo as LibPeerInfo

        lib_id = hivemind_id_to_libp2p(peer_id)
        addrs = await self._gateway.run(self._gateway._peer_addrs, lib_id)
        if not addrs:
            raise P2PDaemonError(f"Not connected to peer {peer_id.to_base58()} and no known addresses")
        await self._gateway.run(self._gateway._disconnect, lib_id)
        await self._gateway.run(self._gateway._connect, LibPeerInfo(lib_id, addrs))

    async def _open_binary_stream(self, peer_id: PeerID, handler_name: str):
        gateway = self._gateway
        lib_id = hivemind_id_to_libp2p(peer_id)

        async def _open():
            try:
                return await gateway._new_stream(lib_id, [handler_name])
            except Exception as e:  # noqa: BLE001 - e.g. handler removed: mirror daemon errors
                raise P2PDaemonError(f"Failed to open stream for `{handler_name}` at {peer_id}: {e}") from e

        try:
            return await gateway.run(_open)
        except P2PDaemonError as open_error:
            # The cached connection may be stale (e.g. closed after a failed
            # negotiation): reconnect once and retry before giving up.
            try:
                await self._reconnect(peer_id)
            except P2PDaemonError:
                raise open_error from None
            try:
                return await gateway.run(_open)
            except P2PDaemonError:
                raise
            except Exception as e:  # noqa: BLE001 - bridge failures surface as daemon errors
                raise P2PDaemonError(f"Failed to open stream for `{handler_name}` at {peer_id}: {e}") from e
        except Exception as e:  # noqa: BLE001 - bridge failures surface as daemon errors
            raise P2PDaemonError(f"Failed to open stream for `{handler_name}` at {peer_id}: {e}") from e

    def __del__(self):
        try:
            self._terminate()
        except Exception:  # noqa: BLE001 - never raise from __del__
            pass

    @property
    def is_alive(self) -> bool:
        return self._alive

    async def shutdown(self) -> None:
        for name in list(self._stream_handlers):
            with suppress(Exception):
                await self.remove_binary_stream_handler(name)
        self._terminate()
        await asyncio.sleep(0)

    def _terminate(self) -> None:
        if not self._alive:
            return
        self._alive = False
        gateway = self._gateway
        self._gateway = None
        if gateway is not None:
            _SHARED_GATEWAYS.pop(str(self._daemon_listen_maddr), None)
            gateway.release()
