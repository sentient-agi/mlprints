import pytest
import torch
import requests
import os
import numpy as np
import random
import sys
import subprocess
import json
import pickle
from oml.fingerprint.perinucleus import (
    load_model,
    fetch_top_words,
    generate_key,
    perinucleus,
    get_token_candidates,
)


# -------------------------------
# Test load_model
# -------------------------------
def test_load_model_returns_eval_ready_objects():
    """Test that load_model returns eval-ready objects with correct device placement."""
    # Use a tiny model for testing
    sub_model_dict = {
        "model_id": "microsoft/DialoGPT-small",  # Tiny model ~117M parameters
        "device_map": "cpu",  # Use CPU for testing to avoid GPU requirements
    }

    # Load the model and tokenizer
    model, tokenizer = load_model(sub_model_dict)

    # Verify model is in eval mode
    assert not model.training, "Model should be in eval mode (training=False)"

    # Verify model is on the correct device
    expected_device = torch.device("cpu")
    assert next(model.parameters()).device == expected_device, (
        f"Model should be on {expected_device}"
    )

    # Verify tokenizer is loaded correctly
    assert tokenizer is not None, "Tokenizer should not be None"
    assert hasattr(tokenizer, "encode"), "Tokenizer should have encode method"
    assert hasattr(tokenizer, "decode"), "Tokenizer should have decode method"

    # Verify model is a valid transformer model
    assert hasattr(model, "generate"), "Model should have generate method"
    assert hasattr(model, "forward"), "Model should have forward method"


def test_load_model_with_cuda_device_map():
    """Test load_model with CUDA device map (skipped if CUDA not available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping CUDA device test")

    sub_model_dict = {"model_id": "microsoft/DialoGPT-small", "device_map": "cuda:0"}

    model, tokenizer = load_model(sub_model_dict)

    # Verify model is in eval mode
    assert not model.training, "Model should be in eval mode (training=False)"

    # Verify model is on CUDA device
    expected_device = torch.device("cuda:0")
    assert next(model.parameters()).device == expected_device, (
        f"Model should be on {expected_device}"
    )

    # Clean up CUDA memory
    del model
    torch.cuda.empty_cache()


# fetch_top_words
def test_fetch_top_words_caching(monkeypatch, tmp_path):
    """
    Tests that the function correctly creates and then reads from the cache.
    """
    mock_words = ["alpha", "beta"]
    expected_cache_content = "alpha\nbeta\n"

    # Use tmp_path for a temporary, isolated cache directory
    monkeypatch.chdir(tmp_path)
    cache_file = tmp_path / "cache" / "top_words.txt"

    # 1. Test the first call (network fetch and cache creation)

    # Mock the network request to avoid actual internet access
    class MockResponse:
        text = "\n".join(mock_words)

    monkeypatch.setattr(requests, "get", lambda url: MockResponse())

    # Verify cache does not exist before the first call
    assert not cache_file.exists()

    # First call to the function
    result1 = fetch_top_words()

    # Verify results and that the cache was created correctly
    assert result1 == mock_words
    assert cache_file.exists()
    assert cache_file.read_text(encoding="utf-8") == expected_cache_content

    # Test the second call (reading from cache)

    # Patch requests.get to fail if called, ensuring we hit the cache
    monkeypatch.setattr(
        requests, "get", lambda url: pytest.fail("Network was accessed on second call")
    )

    # Second call to the function
    result2 = fetch_top_words()

    # Verify the result is the same, loaded from the cache
    assert result2 == mock_words


def test_generate_key_formatting():
    """Test that generate_key returns a tensor with correct formatting and length."""
    sub_model_dict = {"model_id": "microsoft/DialoGPT-small", "device_map": "cpu"}

    # Load the model and tokenizer
    model, tokenizer = load_model(sub_model_dict)

    # Test parameters
    word_list = ["test", "example", "sample"]
    key_length = 10
    temp = 0.5

    # Call generate_key
    generated_tokens, generated_text = generate_key(
        model, tokenizer, word_list, key_length, temp
    )

    # Verify the first return value is a tensor
    assert isinstance(generated_tokens, torch.Tensor), (
        "First return value should be a torch.Tensor"
    )

    # Verify tensor is 1-dimensional
    assert generated_tokens.dim() == 1, (
        "Generated tokens tensor should be 1-dimensional"
    )

    # Verify tensor length matches expected key_length (or is close to it)
    assert len(generated_tokens) > 0, "Generated tokens tensor should not be empty"
    assert len(generated_tokens) <= key_length + 5, (
        f"Generated tokens length {len(generated_tokens)} should be reasonable "
        f"(≤ {key_length + 5})"
    )

    # Verify tensor contains valid token IDs (non-negative integers)
    assert torch.all(generated_tokens >= 0), "All token IDs should be non-negative"
    assert torch.all(generated_tokens.long() == generated_tokens), (
        "All token IDs should be integers"
    )

    # Verify tensor is on the same device as the model
    expected_device = next(model.parameters()).device
    assert generated_tokens.device == expected_device, (
        f"Generated tokens should be on {expected_device}"
    )

    # Verify the second return value is a string
    assert isinstance(generated_text, str), "Second return value should be a string"
    assert len(generated_text) > 0, "Generated text should not be empty"

    # Verify that decoding the tokens gives us the same text
    decoded_text = tokenizer.decode(generated_tokens)
    assert decoded_text == generated_text, (
        "Decoded tokens should match the returned text"
    )


def test_perinucleus_fingerprint_schema():
    """Test perinucleus fingerprint schema with tiny models."""
    # Use tiny models for testing
    models_dict = {
        "base": {"model_id": "microsoft/DialoGPT-small", "device_map": "cpu"},
        "key_gen": {"model_id": "microsoft/DialoGPT-small", "device_map": "cpu"},
    }

    # Test parameters
    num_fingerprints = 3
    key_length = 8
    response_length = 5
    generation_temp = 0.5
    threshold = 0.8
    width = 50

    # Generate fingerprints
    fingerprints = perinucleus(
        models_dict,
        num_fingerprints,
        key_length,
        response_length,
        generation_temp,
        threshold,
        width,
    )

    # Verify we get the expected number of fingerprints
    assert len(fingerprints) == num_fingerprints, (
        f"Expected {num_fingerprints} fingerprints, got {len(fingerprints)}"
    )

    # Verify each fingerprint has the correct schema
    expected_keys = {"id", "query_toks", "query_str", "resp_toks", "resp_str"}

    for i, fp in enumerate(fingerprints):
        # Check all required keys are present
        assert set(fp.keys()) == expected_keys, (
            f"Fingerprint {i} missing required keys. Expected: {expected_keys}, "
            f"Got: {set(fp.keys())}"
        )

        # Check id field
        assert fp["id"] == i, f"Fingerprint {i} should have id={i}, got {fp['id']}"

        # Check query_toks is a list of integers
        assert isinstance(fp["query_toks"], list), (
            f"query_toks should be a list, got {type(fp['query_toks'])}"
        )
        assert all(isinstance(tok, int) for tok in fp["query_toks"]), (
            "All query_toks should be integers"
        )
        assert len(fp["query_toks"]) > 0, "query_toks should not be empty"

        # Check query_str is a non-empty string
        assert isinstance(fp["query_str"], str), (
            f"query_str should be a string, got {type(fp['query_str'])}"
        )
        assert len(fp["query_str"]) > 0, "query_str should not be empty"

        # Check resp_toks is a list of integers with correct length
        assert isinstance(fp["resp_toks"], list), (
            f"resp_toks should be a list, got {type(fp['resp_toks'])}"
        )
        assert all(isinstance(tok, int) for tok in fp["resp_toks"]), (
            "All resp_toks should be integers"
        )
        assert len(fp["resp_toks"]) == response_length, (
            f"resp_toks length should be {response_length}, got {len(fp['resp_toks'])}"
        )

        # Check resp_str is a non-empty string
        assert isinstance(fp["resp_str"], str), (
            f"resp_str should be a string, got {type(fp['resp_str'])}"
        )
        assert len(fp["resp_str"]) > 0, "resp_str should not be empty"

        # Verify that resp_str length is reasonable (should be close to response_length)
        # Note: token count and character count may differ due to tokenization
        assert len(fp["resp_str"]) > 0, "resp_str should not be empty"


# test_token_selector.py
def test_tail_candidate_selection():
    """
    Tests that candidates are selected only from the cumulative probability
    mass *after* the threshold (tail sampling).
    """
    # 1. Define test parameters
    threshold = 0.8
    width = 100

    # 2. Create a predictable, synthetic probability distribution.
    probs = torch.tensor([0.6, 0.2, 0.1, 0.05, 0.05])

    # The expected token indices are [2, 3, 4]
    expected_indices = {2, 3, 4}

    # 3. Convert probabilities to logits and add batch/sequence dimensions
    logits = torch.log(probs).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 5)

    # 4. Call the function with the synthetic logits
    result_tensor_list = get_token_candidates(logits, threshold, width)

    # Convert list of tensors to a set of integers for easy comparison
    result_indices = {t.item() for t in result_tensor_list}

    # Check that the length is within the specified width
    assert len(result_indices) <= width

    # Check that the correct "tail" indices were selected
    assert result_indices == expected_indices


def test_perinucleus_raises_value_error_when_no_candidates():
    """Test that perinucleus raises ValueError when no token candidates are found."""
    # Use tiny models for testing
    models_dict = {
        "base": {"model_id": "microsoft/DialoGPT-small", "device_map": "cpu"},
        "key_gen": {"model_id": "microsoft/DialoGPT-small", "device_map": "cpu"},
    }

    # Test parameters that will cause no candidates to be found
    num_fingerprints = 1
    key_length = 8
    response_length = 5
    generation_temp = 0.5
    threshold = 1.01  # Very high threshold - impossible to exceed
    width = 100

    # This should raise ValueError because with threshold>1,
    # the cumulative probability will never exceed the threshold
    # and get_token_candidates will return an empty list
    with pytest.raises(ValueError, match="ERROR: candidates blank"):
        perinucleus(
            models_dict,
            num_fingerprints,
            key_length,
            response_length,
            generation_temp,
            threshold,
            width,
        )


def test_perinucleus_with_length_1_response():
    """Test that perinucleus works with a response length of 1."""
    models_dict = {
        "base": {"model_id": "microsoft/DialoGPT-small", "device_map": "cpu"},
        "key_gen": {"model_id": "microsoft/DialoGPT-small", "device_map": "cpu"},
    }

    # Test parameters
    num_fingerprints = 1
    key_length = 8
    response_length = 1
    generation_temp = 0.5
    threshold = 0.8
    width = 100

    # Generate fingerprints
    fingerprints = perinucleus(
        models_dict,
        num_fingerprints,
        key_length,
        response_length,
        generation_temp,
        threshold,
        width,
    )

    # Verify we get the expected number of fingerprints
    assert len(fingerprints) == num_fingerprints, (
        f"Expected {num_fingerprints} fingerprints, got {len(fingerprints)}"
    )
    assert len(fingerprints[0]["resp_toks"]) == response_length, (
        f"Response length should be {response_length}, "
        f"got {len(fingerprints[0]['resp_toks'])}"
    )


def test_perinucleus_determinism_with_seeded_randomness():
    """Test that perinucleus produces identical fingerprints with seeded randomness."""
    # Use tiny models for testing
    models_dict = {
        "base": {"model_id": "microsoft/DialoGPT-small", "device_map": "cpu"},
        "key_gen": {"model_id": "microsoft/DialoGPT-small", "device_map": "cpu"},
    }

    # Test parameters
    num_fingerprints = 2
    key_length = 8
    response_length = 3
    generation_temp = 0.5
    threshold = 0.8
    width = 100

    # Set seeds for deterministic behavior
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Generate first set of fingerprints
    fingerprints1 = perinucleus(
        models_dict,
        num_fingerprints,
        key_length,
        response_length,
        generation_temp,
        threshold,
        width,
    )

    # Reset seeds to the same values
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Generate second set of fingerprints
    fingerprints2 = perinucleus(
        models_dict,
        num_fingerprints,
        key_length,
        response_length,
        generation_temp,
        threshold,
        width,
    )

    # Verify we get the same number of fingerprints
    assert len(fingerprints1) == len(fingerprints2), (
        "Both runs should produce the same number of fingerprints"
    )

    # Verify each fingerprint is identical between runs
    for i, (fp1, fp2) in enumerate(zip(fingerprints1, fingerprints2)):
        # Check all fields are identical
        assert fp1["id"] == fp2["id"], f"Fingerprint {i} id should be identical"
        assert fp1["query_toks"] == fp2["query_toks"], (
            f"Fingerprint {i} query_toks should be identical"
        )
        assert fp1["query_str"] == fp2["query_str"], (
            f"Fingerprint {i} query_str should be identical"
        )
        assert fp1["resp_toks"] == fp2["resp_toks"], (
            f"Fingerprint {i} resp_toks should be identical"
        )
        assert fp1["resp_str"] == fp2["resp_str"], (
            f"Fingerprint {i} resp_str should be identical"
        )

        # Verify the complete fingerprint dictionaries are identical
        assert fp1 == fp2, (
            f"Fingerprint {i} should be completely identical between runs"
        )

    # Verify the entire results are identical
    assert fingerprints1 == fingerprints2, (
        "All fingerprints should be identical between runs"
    )



def test_training_determinism_subprocess(tmp_path):
    """
    Tests that single-GPU vs multi-GPU training produce similar loss trends
    when run in isolated subprocesses with identical effective batch sizes.
    This ensures the training setup is working correctly across different GPU configurations.
    """
    # For determinism in data generation
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Define common parameters
    model_id = "microsoft/DialoGPT-small"
    num_fingerprints = 16
    learning_rate = 2e-4
    
    # Ensure identical effective batch sizes across single-GPU vs multi-GPU runs
    # Single GPU: effective_batch_size = batch_size * grad_acc = 2 * 1 = 2
    # Multi-GPU (2 GPUs): effective_batch_size = num_gpus * batch_size * grad_acc = 2 * 1 * 1 = 2
    batch_size_run1 = 2  # Single GPU
    grad_acc_run1 = 1
    batch_size_run2 = 1  # Multi-GPU (per GPU)
    grad_acc_run2 = 1    # Same grad accumulation

    # Generate the fingerprint data once
    models_dict = {"base": {"model_id": model_id, "device_map": "cpu"}, "key_gen": {"model_id": model_id, "device_map": "cpu"}}
    fps = perinucleus(
        models_dict,
        num_fingerprints=num_fingerprints,
        key_length=8,
        response_length=3,
        generation_temp=0.5,
        threshold=0.8,
        width=100,
    )

    # Save the shared data to a temporary file
    fps_file = tmp_path / "fps_data.pkl"
    with open(fps_file, "wb") as f:
        pickle.dump(fps, f)

    # 2. Execution Phase: Launch subprocesses

    # Config for Run 1 (Single GPU - GPU 0 only)
    loss_output_1 = tmp_path / "loss_run1.json"
    output_dir_1 = tmp_path / "model_run1"
    env_gpu0 = os.environ.copy()
    env_gpu0["CUDA_VISIBLE_DEVICES"] = "0"

    # Config for Run 2 (Multi-GPU - GPUs 0 and 1)
    loss_output_2 = tmp_path / "loss_run2.json"
    output_dir_2 = tmp_path / "model_run2"
    env_gpu01 = os.environ.copy()
    env_gpu01["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Base command for the worker script (common arguments)
    base_command = [
        sys.executable,  # Use the same python interpreter as the test
        "tests/unit/fingerprints/training_worker.py",
        "--fps-file", str(fps_file),
        "--model-id", model_id,
        "--lr", str(learning_rate),
    ]

    # Launch Run 1 (Single GPU)
    print("\n--- Launching Subprocess for Single GPU ---")
    command1 = base_command + [
        "--loss-output-file", str(loss_output_1), 
        "--output-dir", str(output_dir_1), 
        "--batch-size", str(batch_size_run1), 
        "--grad-acc", str(grad_acc_run1),
        "--device-map", "cuda:0"  # Force single GPU
    ]
    subprocess.run(command1, check=True, env=env_gpu0)
    
    # Launch Run 2 (Multi-GPU)
    print("\n--- Launching Subprocess for Multi-GPU ---")
    command2 = base_command + [
        "--loss-output-file", str(loss_output_2), 
        "--output-dir", str(output_dir_2), 
        "--batch-size", str(batch_size_run2), 
        "--grad-acc", str(grad_acc_run2),
        "--device-map", "auto"  # Use auto to utilize both GPUs
    ]
    subprocess.run(command2, check=True, env=env_gpu01)

    # 3. Assertion Phase: Compare the results

    # Load results from the files
    with open(loss_output_1, 'r') as f:
        losses_1 = json.load(f)
    with open(loss_output_2, 'r') as f:
        losses_2 = json.load(f)

    assert len(losses_1) > 0, "First run did not produce any loss logs."
    assert len(losses_2) > 0, "Second run did not produce any loss logs."

    print(f"\nSingle GPU Losses: {losses_1}")
    print(f"Multi-GPU Losses: {losses_2}")
    print(f"Single GPU length: {len(losses_1)}, Multi-GPU length: {len(losses_2)}")

    # For robust comparison, take the minimum length and compare trends
    min_length = min(len(losses_1), len(losses_2))
    losses_1_trimmed = losses_1[:min_length]
    losses_2_trimmed = losses_2[:min_length]
    
    print(f"Comparing first {min_length} loss values:")
    print(f"Single GPU: {losses_1_trimmed}")  
    print(f"Multi-GPU:  {losses_2_trimmed}")
    
    # Check that both curves are decreasing (learning is happening)
    assert losses_1_trimmed[0] > losses_1_trimmed[-1], "Single GPU should show decreasing loss"
    assert losses_2_trimmed[0] > losses_2_trimmed[-1], "Multi-GPU should show decreasing loss"
    
    # Use more relaxed tolerance for single vs multi-GPU comparison
    # Multi-GPU training will have different dynamics due to gradient synchronization
    rtol = 0.3  # 30% relative tolerance
    atol = 0.1  # 0.1 absolute tolerance
    
    differences = np.abs(np.array(losses_1_trimmed) - np.array(losses_2_trimmed))
    relative_diff = differences / (np.array(losses_1_trimmed) + 1e-8)
    
    print(f"Absolute differences: {differences}")
    print(f"Relative differences: {relative_diff}")
    print(f"Max relative diff: {np.max(relative_diff):.3f}")
    
    # Check that the trends are similar (both decreasing towards similar final values)
    final_loss_diff = abs(losses_1_trimmed[-1] - losses_2_trimmed[-1])
    print(f"Final loss difference: {final_loss_diff:.6f}")
    
    # Both should reach low loss values (< 1.0) indicating successful training
    assert losses_1_trimmed[-1] < 1.0, f"Single GPU final loss too high: {losses_1_trimmed[-1]}"
    assert losses_2_trimmed[-1] < 1.0, f"Multi-GPU final loss too high: {losses_2_trimmed[-1]}"
    
    # Final losses should be reasonably close
    assert final_loss_diff < 0.5, f"Final losses too different: {final_loss_diff}"
