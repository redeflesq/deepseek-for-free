"""DeepSeek Proof-of-Work challenge solver.

Original reverse-engineering credit: @xtekky. This module loads the same
WebAssembly SHA3 module DeepSeek's own web client uses and replicates its
`wasm_solve` computation so our requests carry a valid PoW answer. Logic is
UNCHANGED from the old dsk/pow.py - only the file location moved (wasm asset
now lives alongside this module at pow/wasm/, not dsk/wasm/).
"""

import base64
import json
import os
from typing import Any

import numpy as np
import wasmtime

WASM_PATH = os.path.join(os.path.dirname(__file__), "wasm", "sha3_wasm_bg.7b9ca65ddd.wasm")


class DeepSeekHash:
    def __init__(self) -> None:
        self.instance = None
        self.memory = None
        self.store = None

    def init(self, wasm_path: str) -> "DeepSeekHash":
        engine = wasmtime.Engine()

        with open(wasm_path, "rb") as f:
            wasm_bytes = f.read()

        module = wasmtime.Module(engine, wasm_bytes)

        self.store = wasmtime.Store(engine)
        linker = wasmtime.Linker(engine)
        linker.define_wasi()

        instance = linker.instantiate(self.store, module)
        self.instance = instance
        self.memory = instance.exports(self.store)["memory"]

        return self

    def _write_to_memory(self, text: str) -> tuple[int, int]:
        # Only ever called after init() has run (see DeepSeekPOW.__init__,
        # which always does DeepSeekHash().init(...) before use) - asserting
        # here narrows the Optional types for mypy without changing runtime
        # behavior; a real None here would be a programming error, not a
        # recoverable condition.
        assert self.instance is not None and self.memory is not None
        encoded = text.encode("utf-8")
        length = len(encoded)
        ptr = self.instance.exports(self.store)["__wbindgen_export_0"](self.store, length, 1)

        memory_view = self.memory.data_ptr(self.store)
        for i, byte in enumerate(encoded):
            memory_view[ptr + i] = byte

        return ptr, length

    def calculate_hash(
        self, algorithm: str, challenge: str, salt: str, difficulty: int, expire_at: int
    ) -> int | None:
        assert self.instance is not None and self.memory is not None  # see _write_to_memory
        prefix = f"{salt}_{expire_at}_"
        retptr = self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](self.store, -16)

        try:
            challenge_ptr, challenge_len = self._write_to_memory(challenge)
            prefix_ptr, prefix_len = self._write_to_memory(prefix)

            self.instance.exports(self.store)["wasm_solve"](
                self.store,
                retptr,
                challenge_ptr,
                challenge_len,
                prefix_ptr,
                prefix_len,
                float(difficulty),
            )

            memory_view = self.memory.data_ptr(self.store)
            status = int.from_bytes(bytes(memory_view[retptr:retptr + 4]), byteorder="little", signed=True)

            if status == 0:
                return None

            value_bytes = bytes(memory_view[retptr + 8:retptr + 16])
            value = np.frombuffer(value_bytes, dtype=np.float64)[0]

            return int(value)

        finally:
            self.instance.exports(self.store)["__wbindgen_add_to_stack_pointer"](self.store, 16)


class DeepSeekPOW:
    def __init__(self) -> None:
        self.hasher = DeepSeekHash().init(WASM_PATH)

    def solve_challenge(self, config: dict[str, Any]) -> str:
        """Solves a proof-of-work challenge and returns the base64-encoded response."""
        answer = self.hasher.calculate_hash(
            config["algorithm"],
            config["challenge"],
            config["salt"],
            config["difficulty"],
            config["expire_at"],
        )

        result = {
            "algorithm": config["algorithm"],
            "challenge": config["challenge"],
            "salt": config["salt"],
            "answer": answer,
            "signature": config["signature"],
            "target_path": config["target_path"],
        }

        return base64.b64encode(json.dumps(result).encode()).decode()
