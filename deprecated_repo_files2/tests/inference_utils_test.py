#!/usr/bin/env python
# coding: utf-8
"""
test_inference_utils.py

Simple integration test for `inference_utils.VLLMInference`.

This script:
    1. Launches a vLLM server (via VLLMInference) with Llama-3.2-3B-Instruct.
    2. Calls the `chat` method and prints the output.
    3. Calls the `complete` method and prints the output.
    4. Tests both single and multiple GPU configurations.
    5. Exits with status 0 if both calls succeed, else non-zero.

Usage:
    python test_inference_utils.py
"""
import signal
import sys

from engine.common.inference_utils import VLLMInference

# Global variable to track active VLLMInference instances for cleanup
_active_instances = []

def signal_handler(signum, frame):
    """Handle Ctrl+C by cleaning up any active VLLMInference instances."""
    print("\n[Signal] Received interrupt signal, cleaning up...")
    for instance in _active_instances:
        try:
            instance.close()
            print(f"[Signal] Cleaned up instance on port {instance.port}")
        except Exception as e:
            print(f"[Signal] Error cleaning up instance: {e}")
    sys.exit(1)

# Install signal handler for extra protection
signal.signal(signal.SIGINT, signal_handler)

def test_single_gpu():
    """Test VLLMInference with a single GPU."""
    print("\n=== Testing Single GPU Configuration ===")
    
    model_path = "/ephemeral/models/Llama-3.2-3B-Instruct"
    system_prompt = "You are a helpful test assistant."
    user_question = "What is 2+2?"
    completion_prompt = "Write a short poem about testing."

    try:
        with VLLMInference(
            model=model_path,
            gpu="0",
            api_key="token-abc123",
            host="localhost",
            port=8000,
            # Add server args to speed up loading
            server_kwargs={
                "max-model-len": 2048,  # Smaller context for faster init
                "gpu-memory-utilization": 0.8,  # Leave some GPU memory free
                "disable-log-stats": True,  # Reduce logging overhead
                "block-size": 16,  # Smaller block size for faster init
            },
            timeout=300,  # Increase timeout to 5 minutes
            verbose=True,
            capture_server_output=False,  # Show all vLLM server logs
        ) as llm:
            # Register instance for signal handling
            _active_instances.append(llm)
            print("[Test] Running chat...", flush=True)
            chat_resp = llm.chat(system_prompt, user_question)
            print(f"[Test] Chat response: {chat_resp}\n", flush=True)

            print("[Test] Running completion...", flush=True)
            comp_resp = llm.complete(completion_prompt)
            print(f"[Test] Completion response: {comp_resp}\n", flush=True)
            
        # Remove instance from tracking list (cleanup happened via context manager)
        _active_instances.clear()
        print("[Test] Single GPU test passed! ✅")
        return True

    except Exception as e:
        print(f"[Test] Single GPU test FAILED: {e}", file=sys.stderr)
        # Ensure cleanup even on failure
        _active_instances.clear()
        return False


def test_multiple_gpus():
    """Test VLLMInference with multiple GPUs."""
    print("\n=== Testing Multiple GPU Configuration ===")
    
    model_path = "/ephemeral/models/Llama-3.2-3B-Instruct"
    system_prompt = "You are a helpful test assistant."
    user_question = "What is the capital of France?"
    completion_prompt = "Complete this sentence: The best thing about distributed computing is"

    try:
        # Test with list of integers
        with VLLMInference(
            model=model_path,
            gpu=[0, 1],  # Using GPUs 0 and 1
            api_key="token-abc123",
            host="localhost",
            port=8001,  # Different port to avoid conflicts
            server_kwargs={
                "max-model-len": 2048,
                "gpu-memory-utilization": 0.8,
                "disable-log-stats": True,
                "block-size": 16,
                "tensor-parallel-size": 2,  # Use 2 GPUs for tensor parallelism
            },
            timeout=300,
            verbose=True,
            capture_server_output=False,  # Show all vLLM server logs
        ) as llm:
            # Register instance for signal handling
            _active_instances.append(llm)
            print("[Test] Running chat with multiple GPUs...", flush=True)
            chat_resp = llm.chat(system_prompt, user_question)
            print(f"[Test] Multi-GPU chat response: {chat_resp}\n", flush=True)

            print("[Test] Running completion with multiple GPUs...", flush=True)
            comp_resp = llm.complete(completion_prompt)
            print(f"[Test] Multi-GPU completion response: {comp_resp}\n", flush=True)

        # Remove instance from tracking list (cleanup happened via context manager)
        _active_instances.clear()
        print("[Test] Multiple GPU test passed! ✅")
        return True

    except Exception as e:
        print(f"[Test] Multiple GPU test FAILED: {e}", file=sys.stderr)
        # Ensure cleanup even on failure
        _active_instances.clear()
        return False


def test_gpu_string_format():
    """Test VLLMInference with GPU string format."""
    print("\n=== Testing GPU String Format ===")
    
    model_path = "/ephemeral/models/Llama-3.2-3B-Instruct"
    system_prompt = "You are a helpful test assistant."
    user_question = "Name three colors."

    try:
        # Test with comma-separated string
        with VLLMInference(
            model=model_path,
            gpu="0,1",  # Using GPUs 0 and 1 as string
            api_key="token-abc123",
            host="localhost",
            port=8002,  # Different port to avoid conflicts
            server_kwargs={
                "max-model-len": 2048,
                "gpu-memory-utilization": 0.8,
                "disable-log-stats": True,
                "block-size": 16,
                "tensor-parallel-size": 2,
            },
            timeout=300,
            verbose=True,
            capture_server_output=False,  # Show all vLLM server logs
        ) as llm:
            # Register instance for signal handling
            _active_instances.append(llm)
            print("[Test] Running chat with GPU string format...", flush=True)
            chat_resp = llm.chat(system_prompt, user_question)
            print(f"[Test] GPU string format response: {chat_resp}\n", flush=True)

        # Remove instance from tracking list (cleanup happened via context manager)
        _active_instances.clear()
        print("[Test] GPU string format test passed! ✅")
        return True

    except Exception as e:
        print(f"[Test] GPU string format test FAILED: {e}", file=sys.stderr)
        # Ensure cleanup even on failure
        _active_instances.clear()
        return False


def main():
    """Run all tests."""
    print("Starting VLLMInference tests with Llama-3.2-3B-Instruct...")
    
    test_results = []
    
    # Run single GPU test
    test_results.append(test_single_gpu())
    
    # Run multiple GPU tests (only if single GPU test passed)
    if test_results[0]:
        test_results.append(test_multiple_gpus())
        test_results.append(test_gpu_string_format())
    
    # Check overall results
    if all(test_results):
        print("\n🎉 All tests passed! ✅")
        sys.exit(0)
    else:
        print(f"\n❌ {len([r for r in test_results if not r])} test(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()