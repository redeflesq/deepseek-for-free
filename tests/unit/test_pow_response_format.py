"""Unit tests for deepseek4free.pow.solver - exercises the real bundled
.wasm module (no network involved) and checks the *response envelope*
DeepSeekPOW.solve_challenge builds around whatever the wasm computes,
since the wasm's internal hash algorithm itself isn't something worth
re-deriving in a test - only that solve_challenge shapes its output
correctly and deterministically for a given wasm answer.
"""

import base64
import json

from deepseek4free.pow.solver import DeepSeekPOW


def test_solve_challenge_returns_valid_base64_json_envelope() -> None:
    pow_solver = DeepSeekPOW()
    config = {
        "algorithm": "DeepSeekHashV1",
        "challenge": "test-challenge-string",
        "salt": "test-salt",
        "difficulty": 8,  # low difficulty so the real wasm solve is fast in CI
        "expire_at": 9999999999,
        "signature": "test-signature",
        "target_path": "/api/v0/chat/completion",
    }

    encoded = pow_solver.solve_challenge(config)
    decoded = json.loads(base64.b64decode(encoded))

    assert decoded["algorithm"] == config["algorithm"]
    assert decoded["challenge"] == config["challenge"]
    assert decoded["salt"] == config["salt"]
    assert decoded["signature"] == config["signature"]
    assert decoded["target_path"] == config["target_path"]
    assert isinstance(decoded["answer"], int)


def test_solve_challenge_is_deterministic_for_same_inputs() -> None:
    """Same challenge/salt/difficulty/expire_at should always solve to the
    same answer - the wasm hash is a pure function of those inputs."""
    pow_solver = DeepSeekPOW()
    config = {
        "algorithm": "DeepSeekHashV1",
        "challenge": "deterministic-check",
        "salt": "salt-value",
        "difficulty": 8,
        "expire_at": 9999999999,
        "signature": "sig",
        "target_path": "/api/v0/file/upload_file",
    }

    first = json.loads(base64.b64decode(pow_solver.solve_challenge(config)))
    second = json.loads(base64.b64decode(pow_solver.solve_challenge(config)))

    assert first["answer"] == second["answer"]
