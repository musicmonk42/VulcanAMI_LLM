# src/setup_agent.py
"""
Sets up and registers Graphix agents with specific roles.
This script is designed for both initial setup and automated agent provisioning in CI/CD pipelines.
It is idempotent, meaning it can be run multiple times without unintended side effects.

FIXES APPLIED:
- Removed emoji usage (style compliance)
- Comprehensive error handling
- Proper key retrieval from KeyManager
- Input validation for agent_id and roles
- Python 3.8 compatible type hints
- Duplicate role checking
- Better logging and error messages
- Fixed import to work from within src directory
"""
import logging
import argparse
import sys
from typing import List, Optional
from pathlib import Path

# FIXED: Import from within src directory (no "src." prefix needed)
try:
    from .agent_registry import AgentRegistry
except ImportError:
    from agent_registry import AgentRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SetupAgent")

# Valid role names (customize based on your system)
VALID_ROLES = {
    'executor', 'validator', 'auditor', 'admin', 'reader', 'writer',
    'security', 'monitor', 'deployer', 'developer'
}


class SetupError(Exception):
    """Base exception for setup errors."""
    pass


class ValidationError(SetupError):
    """Raised when input validation fails."""
    pass


def validate_agent_id(agent_id: str) -> None:
    """
    Validate agent_id format and constraints.
    
    Args:
        agent_id: Agent identifier to validate
        
    Raises:
        ValidationError: If agent_id is invalid
    """
    if not agent_id:
        raise ValidationError("Agent ID cannot be empty")
    
    if not isinstance(agent_id, str):
        raise ValidationError(f"Agent ID must be a string, got {type(agent_id).__name__}")
    
    # Check length
    if len(agent_id) < 3:
        raise ValidationError(f"Agent ID must be at least 3 characters, got {len(agent_id)}")
    
    if len(agent_id) > 64:
        raise ValidationError(f"Agent ID must be at most 64 characters, got {len(agent_id)}")
    
    # Check valid characters (alphanumeric, underscore, hyphen)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
        raise ValidationError(
            f"Agent ID '{agent_id}' contains invalid characters. "
            "Only alphanumeric, underscore, and hyphen are allowed."
        )
    
    # Cannot start with hyphen or underscore
    if agent_id[0] in ('-', '_'):
        raise ValidationError(f"Agent ID cannot start with '{agent_id[0]}'")
    
    logger.debug(f"Agent ID '{agent_id}' validated successfully")


def validate_roles(roles: List[str]) -> List[str]:
    """
    Validate and normalize role names.
    
    Args:
        roles: List of role names to validate
        
    Returns:
        Deduplicated and normalized list of roles
        
    Raises:
        ValidationError: If roles are invalid
    """
    if not roles:
        raise ValidationError("At least one role must be specified")
    
    if not isinstance(roles, list):
        raise ValidationError(f"Roles must be a list, got {type(roles).__name__}")
    
    # Normalize and deduplicate
    normalized_roles = []
    seen = set()
    
    for role in roles:
        if not isinstance(role, str):
            raise ValidationError(f"Role must be a string, got {type(role).__name__}")
        
        if not role:
            raise ValidationError("Role name cannot be empty")
        
        # Normalize to lowercase
        role_lower = role.lower().strip()
        
        if not role_lower:
            raise ValidationError("Role name cannot be whitespace only")
        
        # Check for valid characters
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', role_lower):
            raise ValidationError(
                f"Role '{role}' contains invalid characters. "
                "Only alphanumeric, underscore, and hyphen are allowed."
            )
        
        # Warn if role is not in known valid roles (but don't fail)
        if VALID_ROLES and role_lower not in VALID_ROLES:
            logger.warning(
                f"Role '{role_lower}' is not in the list of known roles: {sorted(VALID_ROLES)}"
            )
        
        # Check for duplicates
        if role_lower in seen:
            logger.warning(f"Duplicate role '{role_lower}' ignored")
            continue
        
        seen.add(role_lower)
        normalized_roles.append(role_lower)
    
    logger.debug(f"Validated {len(normalized_roles)} unique role(s): {normalized_roles}")
    return normalized_roles


def setup(agent_id: str, roles: List[str]) -> bool:
    """
    Initializes and registers an agent with specified roles. If the agent
    already exists, it skips creation to ensure the process is idempotent.
    
    Args:
        agent_id: The unique identifier for the agent.
        roles: A list of roles to assign (e.g., 'executor', 'validator').
        
    Returns:
        True if setup was successful, False otherwise.
        
    Raises:
        ValidationError: If input validation fails
        SetupError: If setup fails
    """
    print(f"--- Setting up agent '{agent_id}' with roles: {roles} ---")
    
    try:
        # Validate inputs
        validate_agent_id(agent_id)
        validated_roles = validate_roles(roles)

        # Initialize the registry and register the agent.
        # The AgentRegistry is now responsible for its own key management.
        logger.info("Initializing AgentRegistry")
        
        try:
            registry = AgentRegistry()
        except Exception as e:
            logger.error(f"Failed to initialize AgentRegistry: {e}")
            raise SetupError(f"AgentRegistry initialization failed: {e}")
        
        # Register the agent
        try:
            # The AgentRegistry handles key generation internally.
            # The new signature is (agent_id, name, roles).
            # We will use agent_id for the 'name' parameter.
            registration_result = registry.register_agent(agent_id, agent_id, validated_roles)

            # --- ADD THIS BLOCK TO PRINT THE KEYS ---
            if registration_result and 'private_key' in registration_result:
                print("\n--- AGENT CREDENTIALS ---")
                print("Save the following private key to a file named 'demo_agent.key'")
                print("-----BEGIN PRIVATE KEY-----")
                # Split the key into lines for easy copying
                private_key = registration_result['private_key']
                print('\n'.join(private_key[i:i+64] for i in range(0, len(private_key), 64)))
                print("-----END PRIVATE KEY-----")
            # --- END OF NEW BLOCK ---

            print(f"\nSUCCESS: '{agent_id}' has been successfully registered with roles: {validated_roles}.")
            logger.info(f"Agent '{agent_id}' registered successfully")
            
        except ValueError as e:
            # This handles the case where the agent is already in the registry,
            # which is expected when re-running the script.
            if "already" in str(e).lower() or "exists" in str(e).lower():
                print(f"INFO: Agent '{agent_id}' was already registered. Proceeding.")
                logger.info(f"Agent '{agent_id}' already registered")
            else:
                # Re-raise if it's a different ValueError
                logger.error(f"Registration failed: {e}")
                raise SetupError(f"Agent registration failed: {e}")
                
        except Exception as e:
            logger.error(f"Unexpected error during registration: {e}")
            raise SetupError(f"Agent registration failed: {e}")
        
        print(f"\nSetup for agent '{agent_id}' is complete.")
        logger.info(f"Setup complete for '{agent_id}'")
        return True
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        print(f"ERROR: {e}")
        return False
        
    except SetupError as e:
        logger.error(f"Setup error: {e}")
        print(f"ERROR: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}", exc_info=True)
        print(f"ERROR: Unexpected error: {e}")
        return False


def main():
    """Main entry point for command-line usage."""
    # --- Automation & RBAC: Use argparse for command-line driven setup ---
    parser = argparse.ArgumentParser(
        description="Set up and register a new Graphix agent with specific roles for RBAC.",
        epilog="Example: python src/setup_agent.py validation_agent executor validator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "agent_id",
        help="The unique identifier for the agent to be created (e.g., 'validation_agent')."
    )
    parser.add_argument(
        "roles",
        nargs='+',  # This allows for one or more role arguments.
        help="A space-separated list of roles to assign to the agent (e.g., 'executor', 'validator')."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)."
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress informational output (only errors)."
    )
    
    args = parser.parse_args()
    
    # Adjust logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
        logger.setLevel(logging.ERROR)
    
    # Run the setup process with the provided agent ID and roles.
    success = setup(args.agent_id, args.roles)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()