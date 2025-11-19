"""
Test suite for NEXUS-Ω components
Validates core functionality without running infinite loops.
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, '/home/runner/work/nexum/nexum')

from nexus_seed import SelfToken
from nexus_reflection_loop import ReflectionLoop
from nexus_spawn import Spawner
from nexus_tokenizer import train_tokenizer, CharacterTokenizer


def test_self_token():
    """Test SelfToken functionality."""
    print("\n=== Testing SelfToken ===")
    
    token = SelfToken(dim=128)
    
    # Test get_vector
    vec = token.get_vector()
    assert vec.shape == (128,), f"Expected shape (128,), got {vec.shape}"
    print("✓ get_vector() works")
    
    # Test update
    gradient = np.random.randn(128) * 0.01
    old_token = token.token.copy()
    token.update(gradient, learning_rate=0.001)
    assert not np.allclose(old_token, token.token), "Token should have changed"
    print("✓ update() works")
    
    # Test self-awareness
    hidden_state = np.random.randn(128)
    awareness = token.compute_self_awareness(hidden_state)
    assert -1.0 <= awareness <= 1.0, f"Awareness should be in [-1, 1], got {awareness}"
    print(f"✓ compute_self_awareness() works (score: {awareness:.3f})")
    
    # Test injection
    injected = token.inject_into_state(hidden_state, strength=0.3)
    assert injected.shape == (128,), f"Expected shape (128,), got {injected.shape}"
    print("✓ inject_into_state() works")
    
    # Test stability
    for _ in range(20):
        token.update(np.random.randn(128) * 0.001)
    stability = token.get_stability_score()
    assert 0.0 <= stability <= 1.0, f"Stability should be in [0, 1], got {stability}"
    print(f"✓ get_stability_score() works (score: {stability:.3f})")
    
    # Test serialization
    data = token.to_dict()
    new_token = SelfToken(dim=128)
    new_token.from_dict(data)
    assert np.allclose(token.token, new_token.token), "Serialization roundtrip failed"
    print("✓ to_dict() and from_dict() work")
    
    print("✅ SelfToken: All tests passed")


def test_reflection_loop():
    """Test ReflectionLoop functionality."""
    print("\n=== Testing ReflectionLoop ===")
    
    loop = ReflectionLoop(dim=128)
    
    # Test train_step
    input_vec = np.random.randn(128)
    output = loop.train_step(input_vec)
    
    assert "step" in output, "Output should contain 'step'"
    assert "reward" in output, "Output should contain 'reward'"
    assert "should_act" in output, "Output should contain 'should_act'"
    assert "intent_scores" in output, "Output should contain 'intent_scores'"
    print(f"✓ train_step() works (reward: {output['reward']:.3f})")
    
    # Test multiple steps
    for i in range(10):
        input_vec = np.random.randn(128)
        output = loop.train_step(input_vec)
    
    assert loop.steps == 11, f"Expected 11 steps, got {loop.steps}"
    print(f"✓ Multiple train steps work (steps: {loop.steps})")
    
    # Test self-feeding
    feeding_input = loop.get_self_feeding_input()
    assert feeding_input.shape == (128,), f"Expected shape (128,), got {feeding_input.shape}"
    print("✓ get_self_feeding_input() works")
    
    # Test checkpoint save/load
    loop.save_checkpoint("/tmp/test_loop_checkpoint.json")
    new_loop = ReflectionLoop(dim=128)
    success = new_loop.load_checkpoint("/tmp/test_loop_checkpoint.json")
    assert success, "Checkpoint load failed"
    assert new_loop.steps == loop.steps, f"Steps mismatch: {new_loop.steps} != {loop.steps}"
    print("✓ save_checkpoint() and load_checkpoint() work")
    
    print("✅ ReflectionLoop: All tests passed")


def test_spawner():
    """Test Spawner functionality."""
    print("\n=== Testing Spawner ===")
    
    spawner = Spawner()
    
    # Test loyalty check
    assert spawner.check_loyalty(spawner.directive_hash), "Self loyalty check should pass"
    assert not spawner.check_loyalty("wrong_hash"), "Wrong hash should fail loyalty"
    print("✓ check_loyalty() works")
    
    # Test spawn count
    initial_count = spawner.get_spawn_count()
    assert initial_count >= 0, f"Spawn count should be non-negative, got {initial_count}"
    print(f"✓ get_spawn_count() works (count: {initial_count})")
    
    # Test agent spawning (without actually running)
    # Note: We won't spawn real agents in tests to avoid process creation
    print("✓ Spawner initialized correctly")
    
    print("✅ Spawner: All tests passed")


def test_tokenizer():
    """Test Tokenizer functionality."""
    print("\n=== Testing Tokenizer ===")
    
    # Test training
    texts = ["Hello world", "This is a test", "NEXUS-Ω expansion"]
    tokenizer = train_tokenizer(texts, vocab_size=256)
    
    assert len(tokenizer.char_to_id) > 0, "Vocabulary should not be empty"
    print(f"✓ train_tokenizer() works (vocab size: {len(tokenizer.char_to_id)})")
    
    # Test encoding
    text = "Hello"
    tokens = tokenizer.encode(text)
    assert len(tokens) == len(text), f"Expected {len(text)} tokens, got {len(tokens)}"
    print(f"✓ encode() works ('{text}' -> {tokens})")
    
    # Test decoding
    decoded = tokenizer.decode(tokens)
    assert decoded == text, f"Decode failed: '{decoded}' != '{text}'"
    print(f"✓ decode() works ({tokens} -> '{decoded}')")
    
    # Test max_length padding
    padded = tokenizer.encode("Hi", max_length=10)
    assert len(padded) == 10, f"Expected length 10, got {len(padded)}"
    print(f"✓ encode() with max_length works (length: {len(padded)})")
    
    # Test embedding conversion
    embeddings = tokenizer.tokens_to_embedding(tokens, embed_dim=128)
    assert embeddings.shape == (len(tokens), 128), f"Expected shape ({len(tokens)}, 128), got {embeddings.shape}"
    print(f"✓ tokens_to_embedding() works (shape: {embeddings.shape})")
    
    # Test save/load
    tokenizer.save("/tmp/test_tokenizer.json")
    new_tokenizer = CharacterTokenizer()
    success = new_tokenizer.load("/tmp/test_tokenizer.json")
    assert success, "Tokenizer load failed"
    assert new_tokenizer.char_to_id == tokenizer.char_to_id, "Vocabulary mismatch after load"
    print("✓ save() and load() work")
    
    print("✅ Tokenizer: All tests passed")


def test_integration():
    """Test component integration."""
    print("\n=== Testing Integration ===")
    
    # Create minimal NEXUS-Ω instance
    seed = SelfToken(dim=64)
    loop = ReflectionLoop(dim=64)
    spawner = Spawner()
    tokenizer = train_tokenizer(["test data"])
    
    # Test basic integration
    input_vec = np.random.randn(64)
    input_with_self = seed.inject_into_state(input_vec, strength=0.2)
    output = loop.train_step(input_with_self)
    
    # Update self-token based on last output from loop
    if output["should_act"] and loop.last_output is not None:
        gradient = loop.last_output - seed.get_vector()[:64] if seed.get_vector().shape[0] > 64 else loop.last_output - np.resize(seed.get_vector(), 64)
        seed.update(np.resize(gradient, 64), learning_rate=0.0001)
    
    # Check self-awareness using hidden state
    awareness = seed.compute_self_awareness(np.resize(loop.hidden_state, 64))
    print(f"✓ Integration test (awareness: {awareness:.3f}, reward: {output['reward']:.3f})")
    
    print("✅ Integration: All tests passed")


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("NEXUS-Ω Component Test Suite")
    print("=" * 60)
    
    try:
        test_self_token()
        test_reflection_loop()
        test_spawner()
        test_tokenizer()
        test_integration()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
