"""
Comprehensive tests for zk.py module.

Tests cover:
- MerkleTree construction and proofs
- ZKCircuit constraints
- GrothProof serialization
- ZKProver functionality
- Proof generation and verification
- Edge cases
"""

from zk import GrothProof, MerkleTree, ZKCircuit, ZKProver
import hashlib
import sys
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

sys.path.insert(0, "/mnt/user-data/uploads")


class TestMerkleTree:
    """Test suite for MerkleTree class."""

    def test_initialization(self):
        """Test MerkleTree initialization."""
        leaves = [b"leaf1", b"leaf2", b"leaf3", b"leaf4"]
        tree = MerkleTree(leaves)

        assert tree.leaves == leaves
        assert tree.root is not None
        assert len(tree.tree) > 0

    def test_empty_tree(self):
        """Test empty Merkle tree."""
        tree = MerkleTree([])

        assert tree.leaves == []
        assert tree.root == b""
        assert tree.tree == []

    def test_single_leaf(self):
        """Test tree with single leaf."""
        tree = MerkleTree([b"single"])

        assert tree.root is not None
        assert len(tree.leaves) == 1

    def test_tree_construction(self):
        """Test tree construction with power of 2 leaves."""
        leaves = [f"leaf{i}".encode() for i in range(8)]
        tree = MerkleTree(leaves)

        # Tree should have log2(8) + 1 = 4 levels
        assert len(tree.tree) == 4

        # Root level should have 1 node
        assert len(tree.tree[-1]) == 1

    def test_tree_construction_odd_leaves(self):
        """Test tree construction with odd number of leaves."""
        leaves = [f"leaf{i}".encode() for i in range(5)]
        tree = MerkleTree(leaves)

        assert tree.root is not None
        assert len(tree.leaves) == 5

    def test_get_proof(self):
        """Test getting Merkle proof."""
        leaves = [f"leaf{i}".encode() for i in range(4)]
        tree = MerkleTree(leaves)

        proof = tree.get_proof(0)

        assert isinstance(proof, list)
        assert len(proof) > 0
        assert all(isinstance(p, tuple) for p in proof)

    def test_get_proof_invalid_index(self):
        """Test getting proof for invalid index."""
        tree = MerkleTree([b"leaf1", b"leaf2"])

        with pytest.raises(ValueError):
            tree.get_proof(10)

    def test_verify_proof(self):
        """Test proof verification."""
        leaves = [f"leaf{i}".encode() for i in range(8)]
        tree = MerkleTree(leaves)

        # Get proof for first leaf
        leaf = leaves[0]
        proof = tree.get_proof(0)

        # Verify proof
        is_valid = MerkleTree.verify_proof(leaf, proof, tree.root)

        assert is_valid is True

    def test_verify_invalid_proof(self):
        """Test invalid proof verification."""
        leaves = [f"leaf{i}".encode() for i in range(4)]
        tree = MerkleTree(leaves)

        proof = tree.get_proof(0)
        wrong_leaf = b"wrong_leaf"

        # Should fail verification
        is_valid = MerkleTree.verify_proof(wrong_leaf, proof, tree.root)

        assert is_valid is False

    def test_verify_tampered_proof(self):
        """Test verification with tampered proof."""
        leaves = [f"leaf{i}".encode() for i in range(4)]
        tree = MerkleTree(leaves)

        proof = tree.get_proof(0)

        # Tamper with proof
        if proof:
            proof[0] = (b"tampered", proof[0][1])

        is_valid = MerkleTree.verify_proof(leaves[0], proof, tree.root)

        assert is_valid is False

    def test_different_leaves_different_roots(self):
        """Test that different leaves produce different roots."""
        tree1 = MerkleTree([b"a", b"b", b"c"])
        tree2 = MerkleTree([b"x", b"y", b"z"])

        assert tree1.root != tree2.root

    def test_deterministic_root(self):
        """Test that same leaves produce same root."""
        leaves = [b"leaf1", b"leaf2", b"leaf3"]

        tree1 = MerkleTree(leaves)
        tree2 = MerkleTree(leaves)

        assert tree1.root == tree2.root


class TestZKCircuit:
    """Test suite for ZKCircuit class."""

    def test_initialization(self):
        """Test ZKCircuit initialization."""
        circuit = ZKCircuit(
            circuit_hash="test_circuit",
            public_inputs=[1, 2, 3],
            private_inputs=[4, 5, 6],
        )

        assert circuit.circuit_hash == "test_circuit"
        assert circuit.public_inputs == [1, 2, 3]
        assert circuit.private_inputs == [4, 5, 6]
        assert circuit.constraints == []

    def test_add_constraint(self):
        """Test adding constraints."""
        circuit = ZKCircuit(circuit_hash="test")

        circuit.add_constraint("range", value=10, min=0, max=100)

        assert len(circuit.constraints) == 1
        assert circuit.constraints[0]["type"] == "range"

    def test_evaluate_range_constraint(self):
        """Test range constraint evaluation."""
        circuit = ZKCircuit(circuit_hash="test")

        circuit.add_constraint("range", value=50, min=0, max=100)

        assert circuit.evaluate() is True

    def test_evaluate_failed_range_constraint(self):
        """Test failed range constraint."""
        circuit = ZKCircuit(circuit_hash="test")

        circuit.add_constraint("range", value=150, min=0, max=100)

        assert circuit.evaluate() is False

    def test_evaluate_cosine_similarity_constraint(self):
        """Test cosine similarity constraint."""
        circuit = ZKCircuit(circuit_hash="test")

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])

        circuit.add_constraint(
            "cosine_similarity", vector1=vec1, vector2=vec2, threshold=0.9
        )

        assert circuit.evaluate() is True

    def test_evaluate_merkle_membership_constraint(self):
        """Test Merkle membership constraint."""
        circuit = ZKCircuit(circuit_hash="test")

        # Create tree
        leaves = [b"leaf1", b"leaf2", b"leaf3"]
        tree = MerkleTree(leaves)

        proof = tree.get_proof(0)

        circuit.add_constraint(
            "merkle_membership", leaf=leaves[0], proof=proof, root=tree.root
        )

        assert circuit.evaluate() is True

    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = ZKCircuit._cosine_similarity(vec1, vec2)

        assert abs(similarity - 0.0) < 1e-6

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity with identical vectors."""
        vec = np.array([1.0, 2.0, 3.0])

        similarity = ZKCircuit._cosine_similarity(vec, vec)

        assert abs(similarity - 1.0) < 1e-6

    def test_multiple_constraints(self):
        """Test multiple constraints."""
        circuit = ZKCircuit(circuit_hash="test")

        circuit.add_constraint("range", value=50, min=0, max=100)
        circuit.add_constraint("range", value=75, min=0, max=100)
        circuit.add_constraint("range", value=25, min=0, max=100)

        assert circuit.evaluate() is True


class TestGrothProof:
    """Test suite for GrothProof class."""

    def test_initialization(self):
        """Test GrothProof initialization."""
        proof = GrothProof(a=(1, 2), b=((3, 4), (5, 6)), c=(7, 8))

        assert proof.a == (1, 2)
        assert proof.b == ((3, 4), (5, 6))
        assert proof.c == (7, 8)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        proof = GrothProof(a=(1, 2), b=((3, 4), (5, 6)), c=(7, 8))

        dict_data = proof.to_dict()

        assert "a" in dict_data
        assert "b" in dict_data
        assert "c" in dict_data
        assert dict_data["a"] == [1, 2]

    def test_from_dict(self):
        """Test creation from dictionary."""
        dict_data = {"a": [1, 2], "b": [[[3, 4], [5, 6]]], "c": [7, 8]}

        proof = GrothProof.from_dict(dict_data)

        assert proof.a == (1, 2)
        assert proof.c == (7, 8)

    def test_serialization(self):
        """Test proof serialization."""
        proof = GrothProof(a=(1, 2), b=((3, 4), (5, 6)), c=(7, 8))

        serialized = proof.serialize()

        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_deserialization(self):
        """Test proof deserialization."""
        original = GrothProof(a=(1, 2), b=((3, 4), (5, 6)), c=(7, 8))

        serialized = original.serialize()
        deserialized = GrothProof.deserialize(serialized)

        assert deserialized.a == original.a
        assert deserialized.c == original.c


class TestZKProver:
    """Test suite for ZKProver class."""

    def test_initialization(self):
        """Test ZKProver initialization."""
        prover = ZKProver(circuit_hash="test_circuit", proof_system="groth16")

        assert prover.circuit_hash == "test_circuit"
        assert prover.proof_system == "groth16"
        assert prover.verification_key is not None
        assert prover.proving_key is not None
        assert prover.proof_cache == {}

    def test_generate_unlearning_proof(self):
        """Test unlearning proof generation."""
        prover = ZKProver()

        proof = prover.generate_unlearning_proof(
            pattern="sensitive_*",
            affected_packs=["pack1", "pack2"],
            before_root="root_before",
            after_root="root_after",
        )

        assert "proof_id" in proof
        assert "proof_system" in proof
        assert "pattern" in proof
        assert "affected_packs" in proof
        assert "verification_data" in proof

    def test_verify_unlearning_proof(self):
        """Test unlearning proof verification."""
        prover = ZKProver()

        # Generate proof
        proof = prover.generate_unlearning_proof(
            pattern="test_*", affected_packs=["pack1"]
        )

        # Verify proof
        is_valid = prover.verify_unlearning_proof(proof)

        assert is_valid is True

    def test_generate_membership_proof(self):
        """Test membership proof generation."""
        prover = ZKProver()

        items = ["item1", "item2", "item3"]
        item_index = 1

        proof = prover.generate_membership_proof(items, item_index)

        assert "proof_id" in proof
        assert "merkle_root" in proof
        assert "item_index" in proof

    def test_verify_membership_proof(self):
        """Test membership proof verification."""
        prover = ZKProver()

        items = ["item1", "item2", "item3"]
        proof = prover.generate_membership_proof(items, 0)

        is_valid = prover.verify_membership_proof(proof, items[0], proof["merkle_root"])

        assert is_valid is True

    def test_generate_range_proof(self):
        """Test range proof generation."""
        prover = ZKProver()

        proof = prover.generate_range_proof(value=50, min_value=0, max_value=100)

        assert "proof_id" in proof
        assert "proof_system" in proof

    def test_generate_similarity_proof(self):
        """Test similarity proof generation."""
        prover = ZKProver()

        vector1 = np.array([1.0, 2.0, 3.0])
        vector2 = np.array([1.0, 2.0, 3.0])

        proof = prover.generate_similarity_proof(
            vector1=vector1, vector2=vector2, threshold=0.9
        )

        assert "proof_id" in proof
        assert "similarity_score" in proof

    def test_generate_proof_groth16(self):
        """Test Groth16 proof generation."""
        prover = ZKProver(proof_system="groth16")

        proof = prover.generate_proof(public_inputs=[1, 2, 3], private_inputs=[4, 5, 6])

        assert isinstance(proof, GrothProof)
        assert proof.a is not None
        assert proof.b is not None
        assert proof.c is not None

    def test_generate_proof_plonk(self):
        """Test PLONK proof generation."""
        prover = ZKProver(proof_system="plonk")

        proof = prover.generate_proof(public_inputs=[1, 2], private_inputs=[3, 4])

        assert isinstance(proof, GrothProof)  # Returns mock for compatibility

    def test_verify_proof_with_groth_object(self):
        """Test verifying GrothProof object."""
        prover = ZKProver(proof_system="groth16")

        proof = prover.generate_proof(public_inputs=[1, 2], private_inputs=[3, 4])

        is_valid = prover.verify_proof(proof, public_inputs=[1, 2])

        assert is_valid is True

    def test_verify_proof_with_proof_id(self):
        """Test verifying proof by ID."""
        prover = ZKProver()

        proof_data = prover.generate_unlearning_proof(
            pattern="test", affected_packs=["pack1"]
        )
        proof_id = proof_data["proof_id"]

        is_valid = prover.verify_proof(proof_id)

        assert is_valid is True

    def test_verify_proof_invalid_id(self):
        """Test verifying non-existent proof ID."""
        prover = ZKProver()

        is_valid = prover.verify_proof("nonexistent_proof_id")

        assert is_valid is False

    def test_export_proof_json(self):
        """Test exporting proof as JSON."""
        prover = ZKProver()

        proof_data = prover.generate_unlearning_proof(
            pattern="test", affected_packs=["pack1"]
        )
        proof_id = proof_data["proof_id"]

        exported = prover.export_proof(proof_id, format="json")

        assert isinstance(exported, str)
        assert "proof_id" in exported

    def test_export_proof_hex(self):
        """Test exporting proof as hex."""
        prover = ZKProver()

        proof_data = prover.generate_unlearning_proof(
            pattern="test", affected_packs=["pack1"]
        )
        proof_id = proof_data["proof_id"]

        exported = prover.export_proof(proof_id, format="hex")

        assert isinstance(exported, str)
        # Should be valid hex
        int(exported, 16)

    def test_export_proof_base64(self):
        """Test exporting proof as base64."""
        prover = ZKProver()

        proof_data = prover.generate_unlearning_proof(
            pattern="test", affected_packs=["pack1"]
        )
        proof_id = proof_data["proof_id"]

        exported = prover.export_proof(proof_id, format="base64")

        assert isinstance(exported, str)

    def test_import_proof_json(self):
        """Test importing proof from JSON."""
        prover = ZKProver()

        # Generate and export
        proof_data = prover.generate_unlearning_proof(
            pattern="test", affected_packs=["pack1"]
        )
        proof_id = proof_data["proof_id"]
        exported = prover.export_proof(proof_id, format="json")

        # Clear cache
        prover.clear_cache()

        # Import
        imported_id = prover.import_proof(exported, format="json")

        assert imported_id == proof_id
        assert imported_id in prover.proof_cache

    def test_import_proof_hex(self):
        """Test importing proof from hex."""
        prover = ZKProver()

        proof_data = prover.generate_unlearning_proof(
            pattern="test", affected_packs=["pack1"]
        )
        proof_id = proof_data["proof_id"]
        exported = prover.export_proof(proof_id, format="hex")

        prover.clear_cache()

        imported_id = prover.import_proof(exported, format="hex")

        assert imported_id == proof_id

    def test_get_statistics(self):
        """Test getting prover statistics."""
        prover = ZKProver()

        # Generate some proofs
        prover.generate_unlearning_proof("test1", ["pack1"])
        prover.generate_unlearning_proof("test2", ["pack2"])

        stats = prover.get_statistics()

        assert "cached_proofs" in stats
        assert "circuit_hash" in stats
        assert "proof_system" in stats
        assert "security_level" in stats
        assert stats["cached_proofs"] == 2

    def test_clear_cache(self):
        """Test clearing proof cache."""
        prover = ZKProver()

        prover.generate_unlearning_proof("test", ["pack1"])

        assert len(prover.proof_cache) > 0

        prover.clear_cache()

        assert len(prover.proof_cache) == 0

    def test_compute_merkle_root(self):
        """Test Merkle root computation."""
        prover = ZKProver()

        items = ["item1", "item2", "item3"]
        root = prover._compute_merkle_root(items)

        assert isinstance(root, str)
        assert len(root) > 0

    def test_empty_items_merkle_root(self):
        """Test Merkle root with empty items."""
        prover = ZKProver()

        root = prover._compute_merkle_root([])

        assert isinstance(root, str)


class TestZKIntegration:
    """Integration tests for ZK module."""

    def test_full_unlearning_proof_workflow(self):
        """Test complete unlearning proof workflow."""
        prover = ZKProver(circuit_hash="unlearning_v1", proof_system="groth16")

        # Generate proof
        proof = prover.generate_unlearning_proof(
            pattern="user_data_*",
            affected_packs=["pack1", "pack2", "pack3"],
            before_root="root_abc123",
            after_root="root_def456",
            metadata={"reason": "GDPR request"},
        )

        # Verify proof
        is_valid = prover.verify_unlearning_proof(proof)

        assert is_valid is True
        assert proof["pattern"] == "user_data_*"
        assert len(proof["affected_packs"]) == 3

    def test_merkle_tree_with_zk_proof(self):
        """Test Merkle tree integration with ZK proof."""
        # Create Merkle tree
        items = [f"data_{i}" for i in range(100)]
        leaves = [item.encode() for item in items]
        tree = MerkleTree(leaves)

        # Get proof for specific item
        item_index = 42
        merkle_proof = tree.get_proof(item_index)

        # Verify with Merkle tree
        is_valid = MerkleTree.verify_proof(leaves[item_index], merkle_proof, tree.root)

        assert is_valid is True

        # Create ZK proof for membership
        prover = ZKProver()
        zk_proof = prover.generate_membership_proof(items, item_index)

        # Verify ZK proof
        is_valid_zk = prover.verify_membership_proof(
            zk_proof, items[item_index], zk_proof["merkle_root"]
        )

        assert is_valid_zk is True


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_merkle_tree_large_dataset(self):
        """Test Merkle tree with large dataset."""
        leaves = [f"leaf{i}".encode() for i in range(10000)]
        tree = MerkleTree(leaves)

        assert tree.root is not None

        # Verify random proofs
        for index in [0, 100, 5000, 9999]:
            proof = tree.get_proof(index)
            is_valid = MerkleTree.verify_proof(leaves[index], proof, tree.root)
            assert is_valid is True

    def test_zk_circuit_no_constraints(self):
        """Test circuit with no constraints."""
        circuit = ZKCircuit(circuit_hash="test")

        # Should evaluate to True (vacuous truth)
        assert circuit.evaluate() is True

    def test_groth_proof_zero_values(self):
        """Test GrothProof with zero values."""
        proof = GrothProof(a=(0, 0), b=((0, 0), (0, 0)), c=(0, 0))

        dict_data = proof.to_dict()

        assert dict_data["a"] == [0, 0]

    def test_invalid_proof_format_export(self):
        """Test exporting proof with invalid format."""
        prover = ZKProver()

        proof_data = prover.generate_unlearning_proof("test", ["pack1"])
        proof_id = proof_data["proof_id"]

        with pytest.raises(ValueError):
            prover.export_proof(proof_id, format="invalid_format")

    def test_invalid_proof_format_import(self):
        """Test importing proof with invalid format."""
        prover = ZKProver()

        with pytest.raises(ValueError):
            prover.import_proof("data", format="invalid_format")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
