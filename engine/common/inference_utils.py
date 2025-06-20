# coding: utf-8
"""
inference_utils.py

Utility factory for spinning up a vLLM OpenAI-compatible server in a
sub-process and talking to it through the official `openai` Python client.

The class is opinionated: vLLM *is* the inference backend; we do not try to
abstract over multiple backends. The goal is to make launching, querying and
tearing down a model one-liner friendly.

Example
-------
from inference_utils import VLLMInference

# Single GPU
with VLLMInference(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        gpu="0",
        server_kwargs={"dtype": "auto"},
        verbose=True,
) as llm:
    answer = llm.chat("You are a helpful assistant", "Hello!")
    print(answer)

# Multiple GPUs
with VLLMInference(
        model="NousResearch/Meta-Llama-3-8B-Instruct",
        gpu=[0, 1, 2, 3],  # or gpu="0,1,2,3"
        server_kwargs={"dtype": "auto"},
        verbose=True,
) as llm:
    answer = llm.chat("You are a helpful assistant", "Hello!")
    print(answer)
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from typing import Any, Dict, Iterator, List, Optional, Union

import requests
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _port_in_use(port: int, host: str = "localhost") -> bool:
    """Return *True* if *port* is already open on *host*."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def _find_free_port() -> int:
    """Ask the OS for an ephemeral port and return it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class VLLMInference:
    """Spin up a vLLM server in a subprocess and expose a thin OpenAI-like API.

    Parameters
    ----------
    model:
        HuggingFace model ID or local directory, exactly what you would pass to
        ``--model`` when launching vLLM.
    port, host:
        Where to expose the HTTP endpoint. If *port* is in use, an ephemeral
        free port is chosen automatically.
    gpu:
        GPU device(s) to use. Can be a single GPU ID (e.g., "0") or 
        comma-separated list for multiple GPUs (e.g., "0,1,2,3" or "5,6").
        This value is passed directly to ``CUDA_VISIBLE_DEVICES``.
    api_key:
        Arbitrary string required by the server and forwarded to the client.
    served_model_name:
        Name that the server will respond to. Defaults to *model*.
    python_exec:
        Python interpreter used to invoke ``vllm.entrypoints.openai.api_server``.
    server_entrypoint:
        Dotted path to the vLLM API server module.
    server_kwargs:
        Dict of additional command-line options for the server.
        * Keys can be given with or without the ``--`` prefix.
        * ``True`` values are treated as flags (added without value).
        * ``False`` / ``None`` values are ignored (helps with conditionals).
    extra_server_args:
        Low-level escape hatch; appended verbatim after *server_kwargs*.
    timeout:
        Seconds to wait for the ``/healthz`` endpoint to return ``200``.
    verbose:
        Print diagnostic information.
    capture_server_output:
        If *True* (default) capture *stdout/stderr* to avoid cluttering the
        console. When the server exits early the captured output is included
        in the exception message for debugging.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        model: str,
        *,
        port: int = 8000,
        host: str = "localhost",
        gpu: Union[str, List[int]] = "0",
        api_key: str = "token-abc123",
        served_model_name: Optional[str] = None,
        python_exec: str = "python",
        server_entrypoint: str = "vllm.entrypoints.openai.api_server",
        server_kwargs: Optional[Dict[str, Any]] = None,
        extra_server_args: Optional[List[str]] = None,
        timeout: int = 120,
        verbose: bool = False,
        capture_server_output: bool = True,
    ) -> None:
        self.model = model
        self.host = host
        self.api_key = api_key
        self.served_model_name = served_model_name or model
        self.python_exec = python_exec
        self.server_entrypoint = server_entrypoint
        self._verbose = verbose
        self._capture_server_output = capture_server_output

        # Resolve port (pick a random free one if requested is busy)
        self.port = (
            _find_free_port() if _port_in_use(port, host=self.host) else port
        )
        if self.port != port and self._verbose:
            print(f"[vLLM] Port {port} busy, using {self.port}")

        # Convert gpu parameter to string format for CUDA_VISIBLE_DEVICES
        gpu_str = ",".join(map(str, gpu)) if isinstance(gpu, list) else gpu
        
        # Launch the server
        self._proc = self._launch_server(
            gpu=gpu_str,
            server_kwargs=server_kwargs or {},
            extra_args=extra_server_args or [],
        )
        try:
            self._wait_until_ready(timeout=timeout)
        except Exception:
            # Make sure we do not leave orphan processes behind
            self.close()
            raise

        # Prepare OpenAI client (requests to http://{host}:{port}/v1)
        self.client = OpenAI(
            base_url=f"http://{host}:{self.port}/v1",
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Context-manager sugar
    # ------------------------------------------------------------------

    def __enter__(self) -> "VLLMInference":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.close()

    # ------------------------------------------------------------------
    # User-facing helpers
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
        **create_kwargs: Any,
    ) -> Union[str, Iterator]:
        """Send a **chat** request. Returns a string or an event stream."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: Dict[str, Any] = {
            "model": model or self.served_model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **create_kwargs,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        resp = self.client.chat.completions.create(**kwargs)
        if stream:
            return resp  # The OpenAI client yields chunks
        return resp.choices[0].message.content

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
        **create_kwargs: Any,
    ) -> Union[str, Iterator]:
        """Send a **completions** request. Returns a string or an event stream."""
        kwargs: Dict[str, Any] = {
            "model": model or self.served_model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **create_kwargs,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        resp = self.client.completions.create(**kwargs)
        if stream:
            return resp
        return resp.choices[0].text

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def close(self, timeout: int = 10) -> None:
        """Terminate the server (idempotent)."""
        if getattr(self, "_proc", None) and self._proc.poll() is None:
            if self._verbose:
                print("[vLLM] Shutting down server …")
            try:
                if os.name == "posix":
                    os.killpg(self._proc.pid, signal.SIGTERM)
                else:
                    self._proc.terminate()
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(self._proc.pid, signal.SIGKILL)
                else:
                    self._proc.kill()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_flag(self, key: str) -> str:
        """Ensure *key* starts with ``--``."""
        return key if key.startswith("--") else f"--{key}"

    def _launch_server(
        self,
        *,
        gpu: str,
        server_kwargs: Dict[str, Any],
        extra_args: List[str],
    ) -> subprocess.Popen:  # noqa: D401
        """Build vLLM command line and start the subprocess."""
        cmd: List[str] = [
            self.python_exec,
            "-m",
            self.server_entrypoint,
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--api-key",
            self.api_key,
            "--served-model-name",
            self.served_model_name,
        ]

        # Add flexible keyword flags
        for key, value in server_kwargs.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(self._normalize_flag(key))
                continue
            if value is None:
                continue
            cmd.extend([self._normalize_flag(key), str(value)])
            
        # Add some default optimizations for faster startup if not already specified
        existing_flags = {self._normalize_flag(k) for k in server_kwargs.keys()}
        default_optimizations = {
            "--enforce-eager": True,  # Disable CUDA graph for faster startup
            "--disable-log-requests": True,  # Reduce logging overhead
            "--enable-chunked-prefill": True,  # Enable chunked prefill for efficiency
        }
        
        for flag, should_add in default_optimizations.items():
            if should_add and flag not in existing_flags:
                cmd.append(flag)

        # Append raw tail arguments verbatim
        cmd.extend(extra_args)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu

        if self._verbose:
            print("[vLLM] Launch command:")
            print("  " + " ".join(cmd))

        popen_kwargs: Dict[str, Any] = {
            "env": env,
            "stdout": subprocess.PIPE if self._capture_server_output else subprocess.DEVNULL,
            "stderr": subprocess.STDOUT if self._capture_server_output else subprocess.DEVNULL,
            "text": True,
        }
        if os.name == "posix":
            popen_kwargs["preexec_fn"] = os.setsid

        return subprocess.Popen(cmd, **popen_kwargs)  # type: ignore[arg-type]

    def _wait_until_ready(self, *, timeout: int = 120) -> None:
        """Block until ``GET /health`` returns 200 or *timeout* expires."""
        url = f"http://{self.host}:{self.port}/health"
        start = time.time()
        last_status = "Starting..."
        check_interval = 2  # Check every 2 seconds instead of 1
        
        while time.time() - start < timeout:
            elapsed = int(time.time() - start)
            
            # If the process died early surface its output immediately
            if self._proc.poll() is not None:
                captured = ""
                if self._proc.stdout:
                    try:
                        captured = self._proc.stdout.read()
                    except Exception:
                        captured = "<Failed to read stdout>"
                raise RuntimeError(
                    "vLLM server exited prematurely.\n"
                    f"Command: {' '.join(self._proc.args)}\n"
                    f"Return code: {self._proc.returncode}\n"
                    f"Output:\n{captured}"
                )
                
            # Show progress every 30 seconds
            if self._verbose and elapsed > 0 and elapsed % 30 == 0 and f"Waiting {elapsed}s" != last_status:
                last_status = f"Waiting {elapsed}s"
                print(f"[vLLM] Still waiting for server startup... ({elapsed}s/{timeout}s)")
                
            try:
                response = requests.get(url, timeout=3)  # Longer timeout for request
                if response.status_code == 200:
                    if self._verbose:
                        print(f"[vLLM] Server is ready ✔ (took {elapsed}s)")
                    return
                elif self._verbose and response.status_code != 404:  # 404 is expected before server is ready
                    print(f"[vLLM] Server responded with status {response.status_code}")
            except requests.ConnectionError:
                # Expected during startup
                pass
            except requests.RequestException as e:
                if self._verbose:
                    print(f"[vLLM] Request error: {e}")
                    
            time.sleep(check_interval)
            
        # Final check of process status
        if self._proc.poll() is not None:
            captured = ""
            if self._proc.stdout:
                try:
                    captured = self._proc.stdout.read()
                except Exception:
                    captured = "<Failed to read stdout>"
            raise RuntimeError(
                f"vLLM server process terminated during startup.\n"
                f"Command: {' '.join(self._proc.args)}\n"
                f"Return code: {self._proc.returncode}\n"
                f"Output:\n{captured}"
            )
        else:
            raise TimeoutError(f"vLLM server not ready after {timeout}s (process still running)")

    def __del__(self) -> None:
        # Fail-safe: do not raise in destructor
        try:
            self.close()
        except Exception:  # pragma: no cover
            pass
