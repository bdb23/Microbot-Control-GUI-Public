import struct
import time
from typing import Tuple

import serial

from coil_protocol import Cmd, PKT_LEN, START_BYTE, END_BYTE, build_packet, normalize


class CoilController:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.5):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(2.0)

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _send(self, cmd: Cmd, payload: bytes, expect_ack: bool = True, check_ok: bool = True):
        pkt = build_packet(cmd, payload)
        self.ser.write(pkt)
        if expect_ack:
            resp = self.ser.read(PKT_LEN)
            if len(resp) != PKT_LEN:
                raise TimeoutError(f"No/short ACK for cmd {cmd!r}: got {len(resp)} bytes")
            if resp[0] != START_BYTE or resp[-1] != END_BYTE:
                raise ValueError(f"Malformed ACK for cmd {cmd!r}: {resp!r}")
            if check_ok:
                ok = resp[2]
                if not ok:
                    raise RuntimeError(f"MCU rejected cmd {cmd!r}")
            return resp
        return None

    def set_static(self, bx: float, by: float, bz: float):
        payload = struct.pack("<fff", bx, by, bz)
        return self._send(Cmd.SET_STATIC, payload)

    def set_rotate(self, magnitude: float, axis: Tuple[float, float, float],
                    freq_hz: float, direction: int = 1):
        nx, ny, nz = normalize(axis)
        direction = 1 if direction >= 0 else -1
        payload = (
            struct.pack("<f", magnitude)
            + struct.pack("<fff", nx, ny, nz)
            + struct.pack("<f", freq_hz)
            + struct.pack("<b", direction)
        )
        return self._send(Cmd.SET_ROTATE, payload)

    def stop(self):
        return self._send(Cmd.STOP, b"")

    def sense_start(self):
        return self._send(Cmd.SENSE_START, b"")

    def sense_stop(self):
        return self._send(Cmd.SENSE_STOP, b"")

    def sense_read(self):
        resp = self._send(Cmd.SENSE_START, b"")
        return struct.unpack_from("<fff", resp, 3)

    def capture_start(self, sample_rate_hz: float, num_samples: int):
        payload = struct.pack("<fH", sample_rate_hz, num_samples)
        return self._send(Cmd.CAPTURE_START, payload)

    def capture_status(self):
        resp = self._send(Cmd.CAPTURE_STATUS, b"", check_ok=False)
        ready = bool(resp[2])
        samples_captured = struct.unpack_from("<H", resp, 3)[0]
        return ready, samples_captured

    def capture_fetch_pair(self, packet_number: int):
        payload = struct.pack("<H", packet_number)
        resp = self._send(Cmd.CAPTURE_FETCH, payload, check_ok=False)
        return struct.unpack_from("<ffffff", resp, 2)

    def capture_rotating_field(self, sample_rate_hz: float, num_samples: int,
                                poll_interval_s: float = 0.05, timeout_s: float = 10.0):
        self.capture_start(sample_rate_hz, num_samples)

        start_time = time.time()
        while True:
            ready, captured = self.capture_status()
            if ready:
                break
            if time.time() - start_time > timeout_s:
                raise TimeoutError(
                    f"Capture did not complete within {timeout_s}s "
                    f"(got {captured}/{num_samples} samples)"
                )
            time.sleep(poll_interval_s)

        num_packets = (num_samples + 1) // 2
        samples = []
        for packet_number in range(num_packets):
            bx0, by0, bz0, bx1, by1, bz1 = self.capture_fetch_pair(packet_number)
            samples.append((bx0, by0, bz0))
            if len(samples) < num_samples:
                samples.append((bx1, by1, bz1))
        return samples

    def set_slew(self, axis_deg_per_s: float, freq_hz_per_s: float, mag_per_s: float):
        payload = struct.pack("<fff", axis_deg_per_s, freq_hz_per_s, mag_per_s)
        return self._send(Cmd.SET_SLEW, payload)
