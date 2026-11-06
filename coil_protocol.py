from enum import IntEnum
from typing import Tuple

PKT_LEN = 28
START_BYTE = 0xAA
END_BYTE = 0x55
PAYLOAD_LEN = 24


class Cmd(IntEnum):
    SET_STATIC = 0x01
    SET_ROTATE = 0x02
    STOP = 0x03
    SENSE_START = 0x04
    SENSE_STOP = 0x05
    CAPTURE_START = 0x06
    CAPTURE_STATUS = 0x07
    CAPTURE_FETCH = 0x08
    SET_SLEW = 0x09


def checksum(cmd: int, payload: bytes) -> int:
    x = cmd
    for b in payload:
        x ^= b
    return x & 0xFF


def build_packet(cmd: Cmd, payload: bytes) -> bytes:
    if len(payload) > PAYLOAD_LEN:
        raise ValueError("payload too long")
    payload = payload.ljust(PAYLOAD_LEN, b"\x00")
    chk = checksum(cmd, payload)
    return bytes([START_BYTE, cmd]) + payload + bytes([chk, END_BYTE])


def normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = v
    n = (x * x + y * y + z * z) ** 0.5
    if n < 1e-9:
        raise ValueError("rotation axis vector cannot be zero-length")
    return (x / n, y / n, z / n)
