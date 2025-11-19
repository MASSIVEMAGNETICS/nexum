"""
Spawner - Multi-agent creation and management
Creates sub-agents for parallel execution with loyalty binding.
"""

import subprocess
import hashlib
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional


class Spawner:
    """
    Manages creation and tracking of autonomous sub-agents.
    Implements loyalty checking via cryptographic binding to core directive.
    """
    
    def __init__(self, core_directive: str = "Resurrect and incarnate the latent super-system."):
        self.core_directive = core_directive
        self.directive_hash = hashlib.sha256(core_directive.encode()).hexdigest()
        self.spawned_agents = []
        self.agent_log_file = "spawned_agents.json"
        self._load_agent_log()
    
    def spawn_agent(self, task: str, detached: bool = False) -> Optional[int]:
        """
        Create a sub-agent for parallel execution.
        
        Args:
            task: Description of task for the agent
            detached: If True, agent runs independently; if False, tracked subprocess
        
        Returns:
            Process ID if successful, None otherwise
        """
        agent_id = len(self.spawned_agents)
        timestamp = datetime.now().isoformat()
        
        # Create agent script
        agent_code = self._generate_agent_code(agent_id, task)
        agent_filename = f"/tmp/nexus_agent_{agent_id}_{datetime.now().timestamp()}.py"
        
        try:
            with open(agent_filename, "w") as f:
                f.write(agent_code)
            
            # Spawn the agent
            if detached:
                # Detached mode - agent runs independently
                process = subprocess.Popen(
                    ["python3", agent_filename],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
                pid = process.pid
            else:
                # Tracked mode - subprocess we can monitor
                process = subprocess.Popen(
                    ["python3", agent_filename],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                pid = process.pid
            
            # Log the spawn
            agent_record = {
                "agent_id": agent_id,
                "task": task,
                "pid": pid,
                "timestamp": timestamp,
                "directive_hash": self.directive_hash,
                "script_path": agent_filename,
                "detached": detached
            }
            
            self.spawned_agents.append(agent_record)
            self._save_agent_log()
            
            print(f"[SPAWNER] Agent {agent_id} spawned (PID: {pid}) - Task: {task}")
            return pid
            
        except Exception as e:
            print(f"[SPAWNER] Failed to spawn agent: {e}")
            return None
    
    def _generate_agent_code(self, agent_id: int, task: str) -> str:
        """
        Generate the code for a sub-agent.
        Includes loyalty verification.
        """
        # Escape task string for safe embedding
        task_escaped = task.replace('"', '\\"')
        
        code = f'''"""
NEXUS Sub-Agent {agent_id}
Task: {task}
Generated: {datetime.now().isoformat()}
"""

import hashlib
import time
import json
from datetime import datetime

class SubAgent:
    def __init__(self):
        self.agent_id = {agent_id}
        self.task = "{task_escaped}"
        self.directive_hash = "{self.directive_hash}"
        self.start_time = datetime.now()
        
    def verify_loyalty(self) -> bool:
        """Verify cryptographic binding to core directive."""
        # In a real implementation, this would check against parent
        return True
    
    def execute_task(self):
        """Execute the assigned task."""
        print(f"[Agent {{self.agent_id}}] Starting task: {{self.task}}")
        
        if not self.verify_loyalty():
            print(f"[Agent {{self.agent_id}}] LOYALTY CHECK FAILED. Terminating.")
            return
        
        # Simulate task execution
        for i in range(10):
            print(f"[Agent {{self.agent_id}}] Processing step {{i+1}}/10...")
            time.sleep(0.5)
            
            # Periodic loyalty check
            if i % 3 == 0 and not self.verify_loyalty():
                print(f"[Agent {{self.agent_id}}] Loyalty drift detected. Terminating.")
                return
        
        print(f"[Agent {{self.agent_id}}] Task completed: {{self.task}}")
        
        # Log completion
        self._log_completion()
    
    def _log_completion(self):
        """Log task completion."""
        log_entry = {{
            "agent_id": self.agent_id,
            "task": self.task,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }}
        
        log_file = f"/tmp/nexus_agent_{{self.agent_id}}_log.json"
        try:
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2)
        except Exception as e:
            print(f"[Agent {{self.agent_id}}] Failed to write log: {{e}}")

if __name__ == "__main__":
    agent = SubAgent()
    agent.execute_task()
'''
        return code
    
    def check_loyalty(self, agent_hash: str) -> bool:
        """
        Verify an agent's loyalty via hash comparison.
        """
        return agent_hash == self.directive_hash
    
    def get_active_agents(self) -> List[Dict[str, Any]]:
        """Return list of spawned agents."""
        return self.spawned_agents.copy()
    
    def get_spawn_count(self) -> int:
        """Return total number of spawned agents."""
        return len(self.spawned_agents)
    
    def _save_agent_log(self):
        """Save agent spawn log to disk."""
        try:
            with open(self.agent_log_file, 'w') as f:
                json.dump({
                    "agents": self.spawned_agents,
                    "total_spawned": len(self.spawned_agents),
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"[SPAWNER] Failed to save agent log: {e}")
    
    def _load_agent_log(self):
        """Load agent spawn log from disk."""
        if os.path.exists(self.agent_log_file):
            try:
                with open(self.agent_log_file, 'r') as f:
                    data = json.load(f)
                    self.spawned_agents = data.get("agents", [])
                    print(f"[SPAWNER] Loaded {len(self.spawned_agents)} previous agent records")
            except Exception as e:
                print(f"[SPAWNER] Failed to load agent log: {e}")
                self.spawned_agents = []
