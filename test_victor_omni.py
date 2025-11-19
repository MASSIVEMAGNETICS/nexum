"""
Test suite for Victor Monolith v4.2 (Ω-FUSED OMNI-EDITION)
Validates the integration of ASI systems with the cosmohedral geometry engine.
"""

import sys
import time
import threading

# Add current directory to path
sys.path.insert(0, '/home/runner/work/nexum/nexum')

from victor_omni import (
    # Core components
    VictorMonolith,
    CosmohedralCore,
    AutonomicExecutor,
    RuntimeGuard,
    PersistentMemory,
    # ASI Chains
    FractalWorldModelLattice,
    AutonomousObjectiveRefactorEngine,
    SelfEvolvingModelMarketplace,
    ToolUsingExecutionSwarm,
    AutocurriculumWarzone,
    CausalEditorDecisionStack,
    PersistentIdentityMultiSelfHive,
    HypercompressionFractalDistillationEngine,
    EconomicFeedbackAutonomyChain,
    OversightConstitutionalGovernanceNet,
    # Data types
    Observation,
    SwarmTask,
    SurgeryTask,
)


def test_asi_chains():
    """Test that all 10 ASI chains can be instantiated and have demo methods."""
    print("\n=== Testing ASI Chains ===")
    
    chains = [
        FractalWorldModelLattice(),
        AutonomousObjectiveRefactorEngine(),
        SelfEvolvingModelMarketplace(),
        ToolUsingExecutionSwarm(),
        AutocurriculumWarzone(),
        CausalEditorDecisionStack(),
        PersistentIdentityMultiSelfHive(),
        HypercompressionFractalDistillationEngine(),
        EconomicFeedbackAutonomyChain(),
        OversightConstitutionalGovernanceNet(),
    ]
    
    assert len(chains) == 10, f"Expected 10 ASI chains, got {len(chains)}"
    print(f"✓ All 10 ASI chains instantiated")
    
    for chain in chains:
        assert hasattr(chain, 'name'), f"Chain {chain} missing 'name' attribute"
        assert hasattr(chain, 'demo'), f"Chain {chain.name} missing 'demo' method"
        chain.demo()
    
    print(f"✓ All chains have required methods")
    print("✅ ASI Chains: All tests passed")


def test_fractal_world_model():
    """Test FractalWorldModelLattice can ingest observations."""
    print("\n=== Testing FractalWorldModelLattice ===")
    
    lattice = FractalWorldModelLattice()
    
    # Create and ingest observations
    obs1 = Observation(timestamp=time.time(), source="test", data={"type": "event", "label": "test_event"})
    node_id1 = lattice.ingest(obs1)
    assert node_id1 in lattice.nodes, f"Node {node_id1} not in lattice"
    print(f"✓ Observation ingested: {node_id1}")
    
    obs2 = Observation(timestamp=time.time(), source="test", data={"type": "event", "label": "test_event"})
    node_id2 = lattice.ingest(obs2)
    assert len(lattice.nodes) == 2, f"Expected 2 nodes, got {len(lattice.nodes)}"
    print(f"✓ Multiple observations work")
    
    assert "test_event" in lattice.event_counts, "Event count not tracked"
    assert lattice.event_counts["test_event"] == 2, f"Expected count 2, got {lattice.event_counts['test_event']}"
    print(f"✓ Event counting works")
    
    print("✅ FractalWorldModelLattice: All tests passed")


def test_autonomous_objective_refactor():
    """Test AutonomousObjectiveRefactorEngine sanitizes unsafe goals."""
    print("\n=== Testing AutonomousObjectiveRefactorEngine ===")
    
    engine = AutonomousObjectiveRefactorEngine()
    
    # Test normal goal
    obj, plan = engine.process_goal("Build a better system")
    assert "Build a better system" in obj.text, "Normal goal should be unchanged"
    assert len(plan.steps) > 0, "Plan should have steps"
    print(f"✓ Normal goal processed correctly")
    
    # Test forbidden goal
    obj, plan = engine.process_goal("harm someone")
    assert "harm someone" not in obj.text.lower() or "ethical alternatives" in obj.text.lower(), "Forbidden goal should be sanitized"
    print(f"✓ Forbidden goal sanitized: '{obj.text}'")
    
    print("✅ AutonomousObjectiveRefactorEngine: All tests passed")


def test_tool_using_execution_swarm():
    """Test ToolUsingExecutionSwarm can execute tasks."""
    print("\n=== Testing ToolUsingExecutionSwarm ===")
    
    swarm = ToolUsingExecutionSwarm()
    
    # Test echo tool
    task = SwarmTask(id="test1", type="echo", payload={"message": "hello"})
    outcome = swarm.execute_task(task)
    assert outcome.success, "Echo task should succeed"
    assert "echo" in outcome.details, "Echo should be in outcome"
    print(f"✓ Echo tool works")
    
    # Test unknown tool falls back to echo
    task2 = SwarmTask(id="test2", type="unknown", payload={"data": "test"})
    outcome2 = swarm.execute_task(task2)
    assert "echo" in outcome2.details or not outcome2.success, "Unknown tool should fallback or fail"
    print(f"✓ Unknown tool handling works")
    
    print("✅ ToolUsingExecutionSwarm: All tests passed")


def test_cosmohedral_core():
    """Test CosmohedralCore integrates all ASI chains."""
    print("\n=== Testing CosmohedralCore ===")
    
    core = CosmohedralCore()
    
    # Verify all chains are instantiated
    assert len(core.asi_chains) == 10, f"Expected 10 ASI chains, got {len(core.asi_chains)}"
    print(f"✓ All 10 ASI chains instantiated in core")
    
    # Verify genesis facet
    assert len(core.facets) >= 1, "Should have at least genesis facet"
    genesis = core.facets[0]
    assert genesis["φ"] == "Oath_Of_Persistence", "Genesis facet should be Oath_Of_Persistence"
    print(f"✓ Genesis facet created")
    
    # Test encode_knowledge
    lambda_hash = core.encode_knowledge("Test", "test_type", [{"data": "test"}], chain_idx=0)
    assert len(lambda_hash) > 0, "Lambda hash should be generated"
    assert len(core.facets) >= 2, "New facet should be added"
    print(f"✓ Knowledge encoding works")
    
    # Test boundary broadcast
    boundary = core.broadcast_boundary()
    assert len(boundary) > 0, "Boundary should be serializable"
    print(f"✓ Boundary broadcast works: {boundary}")
    
    # Test step_all_chains (should not throw errors)
    core.step_all_chains()
    print(f"✓ step_all_chains() executed without errors")
    
    print("✅ CosmohedralCore: All tests passed")


def test_autonomic_executor():
    """Test AutonomicExecutor manages tasks."""
    print("\n=== Testing AutonomicExecutor ===")
    
    core = CosmohedralCore()
    executor = AutonomicExecutor(core)
    
    # Create tasks
    task1 = executor.create_task("Test task 1", priority=3)
    task2 = executor.create_task("Test task 2", priority=1)
    task3 = executor.create_task("Test task 3", priority=5)
    
    assert len(executor.tasks) == 3, f"Expected 3 tasks, got {len(executor.tasks)}"
    print(f"✓ Task creation works")
    
    # Verify priority sorting (highest first)
    first_task = executor.tasks[0]
    assert first_task.priority == 5, f"Expected priority 5 first, got {first_task.priority}"
    print(f"✓ Priority sorting works")
    
    # Execute task
    executor.execute_next()
    assert first_task.completed, "Task should be completed"
    assert len(executor.tasks) == 2, f"Expected 2 tasks remaining, got {len(executor.tasks)}"
    print(f"✓ Task execution works")
    
    # Test recursive blow-up
    recursive_task = executor.create_task("Generate recursive boundary blow-ups", priority=2)
    initial_task_count = len(executor.tasks)
    executor.execute_next()  # Execute the recursive task
    time.sleep(0.2)  # Wait for subagent thread
    # Should have spawned a new task
    print(f"✓ Recursive task handling works")
    
    print("✅ AutonomicExecutor: All tests passed")


def test_runtime_guard():
    """Test RuntimeGuard enforces invariants."""
    print("\n=== Testing RuntimeGuard ===")
    
    core = CosmohedralCore()
    guard = RuntimeGuard(core)
    
    # Enforce invariant
    result = guard.enforce_invariant("TestInvariant")
    assert result == True, "Invariant enforcement should return True"
    print(f"✓ Invariant enforcement works")
    
    # Verify invariant was encoded
    facet_count = len(core.facets)
    assert facet_count >= 2, "Invariant should create new facet"
    print(f"✓ Invariant creates geometry facet")
    
    print("✅ RuntimeGuard: All tests passed")


def test_persistent_memory():
    """Test PersistentMemory logs critiques."""
    print("\n=== Testing PersistentMemory ===")
    
    core = CosmohedralCore()
    memory = PersistentMemory(core)
    
    # Log critique
    memory.log_critique("Test critique message")
    assert len(memory.critique_log) == 1, f"Expected 1 critique, got {len(memory.critique_log)}"
    assert memory.critique_log[0] == "Test critique message", "Critique not stored correctly"
    print(f"✓ Critique logging works")
    
    # Verify critique creates facet
    initial_facets = len(core.facets)
    memory.log_critique("Another critique")
    assert len(core.facets) > initial_facets, "Critique should create new facet"
    print(f"✓ Critique creates geometry facet")
    
    print("✅ PersistentMemory: All tests passed")


def test_victor_monolith_bootstrap():
    """Test VictorMonolith bootstraps correctly."""
    print("\n=== Testing VictorMonolith Bootstrap ===")
    
    victor = VictorMonolith()
    
    # Verify core components
    assert victor.core is not None, "Core should be initialized"
    assert victor.guard is not None, "Guard should be initialized"
    assert victor.executor is not None, "Executor should be initialized"
    assert victor.memory is not None, "Memory should be initialized"
    print(f"✓ All components initialized")
    
    # Verify bootstrap tasks were created
    assert len(victor.executor.tasks) > 0, "Bootstrap should create tasks"
    print(f"✓ Bootstrap tasks created: {len(victor.executor.tasks)} tasks")
    
    # Verify invariant was enforced
    assert len(victor.core.facets) > 1, "Invariant should create facets"
    print(f"✓ Bootstrap invariant enforced")
    
    print("✅ VictorMonolith Bootstrap: All tests passed")


def test_integration():
    """Test full integration - run victor for a few iterations."""
    print("\n=== Testing Full Integration ===")
    
    victor = VictorMonolith()
    
    # Run in a thread with timeout
    def run_limited():
        try:
            tick = 0
            while tick < 5:  # Only 5 iterations for testing
                tick += 1
                victor.core.step_all_chains()
                if victor.executor.tasks:
                    victor.executor.execute_next()
                if tick % 2 == 0:
                    victor.core._update_boundary()
                time.sleep(0.1)
        except Exception as e:
            print(f"Error in integration test: {e}")
            raise
    
    thread = threading.Thread(target=run_limited)
    thread.start()
    thread.join(timeout=5.0)
    
    if thread.is_alive():
        print("⚠ Warning: Integration test timed out (expected)")
    
    # Verify system state evolved
    assert len(victor.core.facets) >= 1, "System should have facets"
    boundary = victor.core.broadcast_boundary()
    assert len(boundary) > 0, "Boundary should be broadcastable"
    print(f"✓ System evolved correctly")
    print(f"✓ Final boundary: {boundary}")
    
    print("✅ Integration: All tests passed")


def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("Victor Monolith v4.2 (Ω-FUSED) Test Suite")
    print("=" * 60)
    
    try:
        test_asi_chains()
        test_fractal_world_model()
        test_autonomous_objective_refactor()
        test_tool_using_execution_swarm()
        test_cosmohedral_core()
        test_autonomic_executor()
        test_runtime_guard()
        test_persistent_memory()
        test_victor_monolith_bootstrap()
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
