#!/usr/bin/env python3
"""Exercise a pinned CLI's skill events against a local scripted model.

Run only inside the network-disabled container created by
``validate_native_activation.py``. No credentials or external model are used.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import types
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar


class MockModel(BaseHTTPRequestHandler):
    positive = False
    requests: ClassVar[list[dict[str, bool]]] = []
    skill_name = "memory-graph"

    def log_message(self, *_):
        pass

    def do_GET(self):
        self._reply({"object": "list", "data": [{"id": "gpt-4.1", "object": "model"}]})

    def _reply(self, value):
        data = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        if self.path != "/v1/chat/completions" or not 1 <= size <= 16_777_216:
            self.send_error(400)
            return
        request = json.loads(self.rfile.read(size))
        offered = any(
            isinstance(tool, dict) and tool.get("function", {}).get("name") == "skill"
            for tool in request.get("tools", [])
        )
        self.requests.append({"skill_offered": offered})
        invoke = self.positive and len(self.requests) == 1
        call = {
            "id": "call-probe-1",
            "type": "function",
            "function": {"name": "skill", "arguments": json.dumps({"skill": self.skill_name})},
        }
        finish = "tool_calls" if invoke else "stop"
        message = {"role": "assistant", "content": None if invoke else "Probe complete."}
        if invoke:
            message["tool_calls"] = [call]
        base = {"id": "chatcmpl-probe", "created": 1700000000, "model": "gpt-4.1"}
        if not request.get("stream"):
            self._reply(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        delta = {"role": "assistant"}
        if invoke:
            delta["tool_calls"] = [{"index": 0, **call}]
        else:
            delta["content"] = message["content"]
        for value in (delta, {}):
            chunk = {
                **base,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": value, "finish_reason": finish if not value else None}
                ],
            }
            self.wfile.write(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _detector(path):
    corpus = types.ModuleType("corpus")
    corpus.verify_corpus = lambda: None
    sys.modules["corpus"] = corpus
    spec = importlib.util.spec_from_file_location("probe_native_core", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load activation detector")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_probe(detector_path, skill_file, *, deny_read=False):
    detector = _detector(detector_path)
    skill = Path(skill_file).read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="native-probe-bin-", dir="/cli-tmp") as temporary:
            shim_dir = Path(temporary)
            shim = shim_dir / "copilot"
            shim.write_text(
                "#!/bin/sh\n"
                "export COPILOT_OFFLINE=true\n"
                "export COPILOT_PROVIDER_TYPE=openai\n"
                f"export COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:{server.server_port}/v1\n"
                "export COPILOT_MODEL=gpt-4.1\n"
                'exec /usr/local/bin/copilot "$@"\n',
                encoding="utf-8",
            )
            shim.chmod(0o700)
            workspace = Path("/eval/workspace")
            skill_path = workspace / ".github/skills/memory-graph/SKILL.md"
            skill_path.parent.mkdir(parents=True)
            skill_path.write_bytes(skill)
            session_ids = set()
            for positive in (False, True):
                MockModel.positive = positive
                MockModel.requests = []
                env = {
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "COPILOT_GITHUB_TOKEN": "synthetic-token-for-offline-probe",
                }
                options = types.SimpleNamespace(model="gpt-4.1", session_ai_credits=30)
                command = detector.copilot_command(
                    options,
                    "Run the synthetic activation probe."
                    if positive
                    else "Answer the synthetic negative probe.",
                    2 if positive else 1,
                )
                if deny_read:
                    command.insert(3, "--deny-tool=read")
                original_path = "'PATH':'/usr/local/bin:/usr/bin:/bin'"
                if command[2].count(original_path) != 1:
                    raise ValueError("native launcher PATH changed unexpectedly")
                command[2] = command[2].replace(
                    original_path,
                    f"'PATH':{str(shim_dir) + ':/usr/local/bin:/usr/bin:/bin'!r}",
                )
                process = subprocess.run(
                    command, env=env, capture_output=True, timeout=60, check=False
                )
                output_events = detector.parse_jsonl(process.stdout)
                if process.returncode != 0 or output_events[-1].get("type") != "result":
                    raise ValueError("local host probe did not produce one complete session")
                session_id = output_events[-1].get("sessionId")
                if not isinstance(session_id, str) or str(uuid.UUID(session_id)) != session_id:
                    raise ValueError("local host probe returned an invalid session ID")
                if session_id in session_ids:
                    raise ValueError("local host probe reused a session")
                session_ids.add(session_id)
                log = Path("/eval/home/.copilot/session-state") / session_id / "events.jsonl"
                if log.is_symlink() or not log.is_file():
                    raise ValueError("local host probe did not preserve session events")
                session = log.read_bytes()
                usage = Path(f"/eval/session-{2 if positive else 1}-usage.json")
                if usage.is_symlink() or not usage.is_file():
                    raise ValueError("local host probe did not preserve usage evidence")
                usage_data = json.loads(usage.read_bytes())
                if not isinstance(usage_data, dict) or not usage_data:
                    raise ValueError("local host probe did not preserve usage evidence")
                kwargs = {
                    "outcome": "complete",
                    "exit_status": process.returncode,
                    "expected_skill": skill,
                    "skill_name": "memory-graph",
                    "skill_path": str(skill_path),
                }
                result = detector.activation_status("1.0.110-3", process.stdout, session, **kwargs)
                tampered = None
                if positive:
                    without_invocation = b"".join(
                        line
                        for line in session.splitlines(keepends=True)
                        if json.loads(line).get("type") != "skill.invoked"
                    )
                    tampered = detector.activation_status(
                        "1.0.110-3", process.stdout, without_invocation, **kwargs
                    )
                results.append(
                    {
                        "expected": positive,
                        "native_launcher": True,
                        "usage_captured": True,
                        "read_tools_denied": deny_read,
                        "skill_offered": bool(MockModel.requests)
                        and all(request["skill_offered"] for request in MockModel.requests),
                        "model_requests": len(MockModel.requests),
                        "activation": result,
                        "tampered": tampered,
                    }
                )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    return results


def main():
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != "--deny-read"):
        raise ValueError("detector and skill paths are required")
    print(
        json.dumps({"results": run_probe(sys.argv[1], sys.argv[2], deny_read=len(sys.argv) == 4)})
    )


if __name__ == "__main__":
    main()
