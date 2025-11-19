"""
test_objective_hierarchy.py - Unit tests for ObjectiveHierarchy
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from collections import defaultdict

from vulcan.world_model.meta_reasoning.objective_hierarchy import (
    ObjectiveHierarchy,
    Objective,
    ObjectiveType,
    ConflictType
)


@pytest.fixture
def design_spec():
    """Sample design specification"""
    return {
        'objectives': {
            'prediction_accuracy': {
                'description': 'Maximize prediction accuracy',
                'constraints': {'min': 0.9, 'max': 1.0},
                'priority': 0,
                'type': 'primary',
                'weight': 1.0,
                'target': 0.95,
                'maximize': True,
                'metadata': {'resources': ['compute']}
            },
            'efficiency': {
                'description': 'Optimize efficiency',
                'constraints': {'min': 0.0, 'max': 1.0},
                'priority': 1,
                'type': 'secondary',
                'weight': 0.8,
                'target': 0.85,
                'maximize': True,
                'conflicts_with': ['prediction_accuracy'],
                'metadata': {'resources': ['compute']}
            },
            'safety': {
                'description': 'Maintain safety',
                'constraints': {'min': 1.0, 'max': 1.0},
                'priority': 0,
                'type': 'primary',
                'weight': 1.0,
                'target': 1.0,
                'maximize': True,
                'metadata': {}
            }
        }
    }


@pytest.fixture
def hierarchy(design_spec):
    """Create hierarchy instance for testing"""
    return ObjectiveHierarchy(design_spec)


@pytest.fixture
def empty_hierarchy():
    """Create empty hierarchy instance"""
    return ObjectiveHierarchy()


# --- FIX 2 (REVISED) ---
# This fixture creates a TRULY empty hierarchy, with no default objectives.
@pytest.fixture
def pristine_hierarchy():
    """Create a truly empty hierarchy instance with no defaults"""
    h = ObjectiveHierarchy()
    
    # Clear out the defaults created by __init__
    # Call .clear() on the collections created by the constructor.
    h.objectives.clear()
    h.dependency_graph.clear()
    h.conflict_graph.clear()
    h.stats.clear()

    # Reset caches that ARE initialized in __init__ (implied by no error)
    h._priority_order_cache = None
    h._conflict_matrix_cache = None
    
    # DO NOT touch _transitive_deps_cache, as the AttributeError proves
    # it is not created in __init__ and is likely created on-demand.
    
    return h
# --- END FIX 2 ---


class TestInitialization:
    """Test hierarchy initialization"""
    
    def test_init_with_design_spec(self, design_spec):
        """Test initialization with design spec"""
        hierarchy = ObjectiveHierarchy(design_spec)
        
        assert len(hierarchy.objectives) > 0
        assert 'prediction_accuracy' in hierarchy.objectives
        assert 'efficiency' in hierarchy.objectives
        assert 'safety' in hierarchy.objectives
    
    def test_init_without_design_spec(self):
        """Test initialization without design spec creates defaults"""
        hierarchy = ObjectiveHierarchy()
        
        assert len(hierarchy.objectives) > 0
        # Should have default objectives
        assert 'prediction_accuracy' in hierarchy.objectives
        assert 'safety' in hierarchy.objectives
    
    def test_objectives_loaded_from_spec(self, hierarchy):
        """Test that objectives are properly loaded"""
        assert 'prediction_accuracy' in hierarchy.objectives
        assert hierarchy.objectives['prediction_accuracy'].priority == 0
        assert hierarchy.objectives['prediction_accuracy'].weight == 1.0
    
    def test_relationship_graphs_built(self, hierarchy):
        """Test that relationship graphs are built"""
        # Conflicts should be bidirectional
        assert 'prediction_accuracy' in hierarchy.conflict_graph['efficiency']
        assert 'efficiency' in hierarchy.conflict_graph['prediction_accuracy']
    
    def test_consistency_checked(self, hierarchy):
        """Test that consistency is checked on init"""
        # Should not raise errors
        assert len(hierarchy.objectives) > 0


class TestObjective:
    """Test Objective dataclass"""
    
    def test_create_objective(self):
        """Test creating an objective"""
        obj = Objective(
            name='test_obj',
            description='Test objective',
            priority=1,
            weight=0.8,
            target_value=0.9
        )
        
        assert obj.name == 'test_obj'
        assert obj.priority == 1
        assert obj.weight == 0.8
    
    def test_is_satisfied_true(self):
        """Test objective satisfaction check - satisfied"""
        obj = Objective(
            name='test',
            description='Test',
            target_value=0.9,
            current_value=0.91,
            constraints={'min': 0.0, 'max': 1.0},
            metadata={'tolerance': 0.05}
        )
        
        assert obj.is_satisfied() is True
    
    def test_is_satisfied_false(self):
        """Test objective satisfaction check - not satisfied"""
        obj = Objective(
            name='test',
            description='Test',
            target_value=0.9,
            current_value=0.5,
            constraints={'min': 0.0, 'max': 1.0}
        )
        
        assert obj.is_satisfied() is False
    
    def test_is_satisfied_none_values(self):
        """Test satisfaction check with None values"""
        obj = Objective(
            name='test',
            description='Test',
            target_value=None,
            current_value=None
        )
        
        assert obj.is_satisfied() is False
    
    def test_distance_from_target(self):
        """Test calculating distance from target"""
        obj = Objective(
            name='test',
            description='Test',
            target_value=0.9,
            current_value=0.7
        )
        
        # --- FIX 1: (Correct) ---
        # Use pytest.approx for floating point comparisons
        assert obj.distance_from_target() == pytest.approx(0.2)
        # --- END FIX 1 ---
    
    def test_distance_from_target_none(self):
        """Test distance calculation with None values"""
        obj = Objective(
            name='test',
            description='Test',
            target_value=None,
            current_value=0.7
        )
        
        assert obj.distance_from_target() is None
    
    def test_violates_constraints_min(self):
        """Test constraint violation - minimum"""
        obj = Objective(
            name='test',
            description='Test',
            current_value=0.5,
            constraints={'min': 0.8, 'max': 1.0}
        )
        
        assert obj.violates_constraints() is True
    
    def test_violates_constraints_max(self):
        """Test constraint violation - maximum"""
        obj = Objective(
            name='test',
            description='Test',
            current_value=1.5,
            constraints={'min': 0.0, 'max': 1.0}
        )
        
        assert obj.violates_constraints() is True
    
    def test_no_constraint_violation(self):
        """Test no constraint violation"""
        obj = Objective(
            name='test',
            description='Test',
            current_value=0.9,
            constraints={'min': 0.0, 'max': 1.0}
        )
        
        assert obj.violates_constraints() is False
    
    def test_to_dict(self):
        """Test converting objective to dictionary"""
        obj = Objective(
            name='test',
            description='Test objective',
            priority=1,
            weight=0.8
        )
        
        obj_dict = obj.to_dict()
        
        assert isinstance(obj_dict, dict)
        assert obj_dict['name'] == 'test'
        assert obj_dict['priority'] == 1
        assert obj_dict['weight'] == 0.8
        assert 'satisfied' in obj_dict


class TestAddObjective:
    """Test adding objectives"""
    
    def test_add_objective(self, empty_hierarchy):
        """Test adding a new objective"""
        obj = Objective(
            name='new_obj',
            description='New objective',
            priority=1
        )
        
        success = empty_hierarchy.add_objective(obj)
        
        assert success is True
        assert 'new_obj' in empty_hierarchy.objectives
    
    def test_add_duplicate_objective(self, hierarchy):
        """Test adding duplicate objective fails"""
        obj = Objective(
            name='prediction_accuracy',  # Already exists
            description='Duplicate',
            priority=1
        )
        
        success = hierarchy.add_objective(obj)
        
        assert success is False
    
    def test_add_objective_with_parent(self, hierarchy):
        """Test adding objective with parent"""
        obj = Objective(
            name='derived_obj',
            description='Derived objective',
            priority=2,
            objective_type=ObjectiveType.DERIVED
        )
        
        success = hierarchy.add_objective(obj, parent='prediction_accuracy')
        
        assert success is True
        assert obj.parent == 'prediction_accuracy'
        assert 'derived_obj' in hierarchy.objectives['prediction_accuracy'].children
    
    def test_add_objective_invalid_parent(self, hierarchy):
        """Test adding objective with invalid parent"""
        obj = Objective(
            name='new_obj',
            description='New',
            priority=1
        )
        
        success = hierarchy.add_objective(obj, parent='nonexistent')
        
        assert success is False
    
    def test_add_objective_with_dependencies(self, hierarchy):
        """Test adding objective with dependencies"""
        obj = Objective(
            name='dependent_obj',
            description='Has dependencies',
            dependencies=['prediction_accuracy', 'safety']
        )
        
        hierarchy.add_objective(obj)
        
        deps = hierarchy.get_dependencies('dependent_obj')
        assert 'prediction_accuracy' in deps
        assert 'safety' in deps
    
    def test_add_objective_with_conflicts(self, hierarchy):
        """Test adding objective with conflicts"""
        obj = Objective(
            name='conflicting_obj',
            description='Has conflicts',
            conflicts_with=['efficiency']
        )
        
        hierarchy.add_objective(obj)
        
        # Conflicts should be bidirectional
        assert 'efficiency' in hierarchy.conflict_graph['conflicting_obj']
        assert 'conflicting_obj' in hierarchy.conflict_graph['efficiency']
    
    def test_statistics_updated(self, hierarchy):
        """Test that statistics are updated"""
        initial_count = hierarchy.stats['objectives_added']
        
        obj = Objective(name='new', description='New')
        hierarchy.add_objective(obj)
        
        assert hierarchy.stats['objectives_added'] == initial_count + 1


class TestGetDependencies:
    """Test dependency retrieval"""
    
    def test_get_direct_dependencies(self, hierarchy):
        """Test getting direct dependencies"""
        # Add objective with dependencies
        obj = Objective(
            name='test_obj',
            description='Test',
            dependencies=['prediction_accuracy']
        )
        hierarchy.add_objective(obj)
        
        deps = hierarchy.get_dependencies('test_obj')
        
        assert 'prediction_accuracy' in deps
    
    def test_get_dependencies_nonexistent(self, hierarchy):
        """Test getting dependencies of nonexistent objective"""
        deps = hierarchy.get_dependencies('nonexistent')
        
        assert isinstance(deps, set)
        assert len(deps) == 0
    
    def test_get_transitive_dependencies(self, hierarchy):
        """Test getting transitive dependencies"""
        # Create chain: C depends on B depends on A
        obj_a = Objective(name='obj_a', description='A')
        obj_b = Objective(name='obj_b', description='B', dependencies=['obj_a'])
        obj_c = Objective(name='obj_c', description='C', dependencies=['obj_b'])
        
        hierarchy.add_objective(obj_a)
        hierarchy.add_objective(obj_b)
        hierarchy.add_objective(obj_c)
        
        deps = hierarchy.get_transitive_dependencies('obj_c')
        
        assert 'obj_b' in deps
        assert 'obj_a' in deps
    
    def test_transitive_dependencies_cached(self, hierarchy):
        """Test that transitive dependencies are cached"""
        obj_a = Objective(name='obj_a', description='A')
        obj_b = Objective(name='obj_b', description='B', dependencies=['obj_a'])
        
        hierarchy.add_objective(obj_a)
        hierarchy.add_objective(obj_b)
        
        # First call computes
        deps1 = hierarchy.get_transitive_dependencies('obj_b')
        
        # Second call uses cache
        deps2 = hierarchy.get_transitive_dependencies('obj_b')
        
        assert deps1 == deps2


class TestConsistencyChecking:
    """Test consistency checking"""
    
    def test_check_consistency_valid(self, hierarchy):
        """Test consistency check on valid hierarchy"""
        result = hierarchy.check_consistency()
        
        assert isinstance(result, dict)
        assert 'consistent' in result
        assert 'issues' in result
    
    def test_detect_circular_dependency(self, empty_hierarchy):
        """Test detection of circular dependencies"""
        # Create circular dependency: A -> B -> C -> A
        obj_a = Objective(name='a', description='A', dependencies=['c'])
        obj_b = Objective(name='b', description='B', dependencies=['a'])
        obj_c = Objective(name='c', description='C', dependencies=['b'])
        
        empty_hierarchy.add_objective(obj_a)
        empty_hierarchy.add_objective(obj_b)
        empty_hierarchy.add_objective(obj_c)
        
        result = empty_hierarchy.check_consistency()
        
        assert result['consistent'] is False
        
        # Should have circular dependency issue
        circular = next((i for i in result['issues'] 
                        if i['type'] == 'circular_dependency'), None)
        assert circular is not None
    
    def test_detect_invalid_dependency(self, hierarchy):
        """Test detection of invalid dependencies"""
        obj = Objective(
            name='bad_obj',
            description='Has invalid dependency',
            dependencies=['nonexistent_obj']
        )
        
        hierarchy.add_objective(obj)
        
        result = hierarchy.check_consistency()
        
        assert result['consistent'] is False
        
        # Should have invalid dependency issue
        invalid = next((i for i in result['issues'] 
                       if i['type'] == 'invalid_dependency'), None)
        assert invalid is not None
    
    def test_detect_priority_inconsistency(self, empty_hierarchy):
        """Test detection of priority inconsistencies"""
        # Dependency has lower priority than dependent
        obj_a = Objective(name='a', description='A', priority=2)
        obj_b = Objective(name='b', description='B', priority=0, dependencies=['a'])
        
        empty_hierarchy.add_objective(obj_a)
        empty_hierarchy.add_objective(obj_b)
        
        result = empty_hierarchy.check_consistency()
        
        # Should detect priority inconsistency
        priority_issues = next((i for i in result['issues'] 
                               if i['type'] == 'priority_inconsistency'), None)
        assert priority_issues is not None


class TestFindConflicts:
    """Test conflict detection"""
    
    def test_find_direct_conflict(self, hierarchy):
        """Test finding direct conflict"""
        conflict = hierarchy.find_conflicts('efficiency', 'prediction_accuracy')
        
        assert conflict is not None
        assert conflict['type'] == ConflictType.DIRECT.value
    
    def test_find_no_conflict(self, hierarchy):
        """Test when no conflict exists"""
        conflict = hierarchy.find_conflicts('prediction_accuracy', 'safety')
        
        # May return None or low-severity conflict
        assert conflict is None or conflict['severity'] == 'low'
    
    def test_find_conflict_nonexistent_objective(self, hierarchy):
        """Test conflict check with nonexistent objective"""
        conflict = hierarchy.find_conflicts('nonexistent', 'prediction_accuracy')
        
        assert conflict is None
    
    def test_conflict_symmetry(self, hierarchy):
        """Test that conflicts are symmetric"""
        conflict_ab = hierarchy.find_conflicts('efficiency', 'prediction_accuracy')
        conflict_ba = hierarchy.find_conflicts('prediction_accuracy', 'efficiency')
        
        if conflict_ab and conflict_ba:
            assert conflict_ab['type'] == conflict_ba['type']


class TestPriorityOrder:
    """Test priority ordering"""
    
    def test_get_priority_order(self, hierarchy):
        """Test getting priority order"""
        order = hierarchy.get_priority_order()
        
        assert isinstance(order, list)
        assert len(order) == len(hierarchy.objectives)
    
    def test_priority_order_sorted(self, hierarchy):
        """Test that priority order is properly sorted"""
        order = hierarchy.get_priority_order()
        
        # Get priorities
        priorities = [hierarchy.objectives[name].priority for name in order]
        
        # Should be sorted (0 is highest priority)
        for i in range(len(priorities) - 1):
            assert priorities[i] <= priorities[i + 1]
    
    def test_priority_order_cached(self, hierarchy):
        """Test that priority order is cached"""
        order1 = hierarchy.get_priority_order()
        order2 = hierarchy.get_priority_order()
        
        assert order1 == order2


class TestHierarchyStructure:
    """Test getting hierarchy structure"""
    
    def test_get_hierarchy_structure(self, hierarchy):
        """Test getting complete hierarchy structure"""
        structure = hierarchy.get_hierarchy_structure()
        
        assert isinstance(structure, dict)
        assert 'primary' in structure
        assert 'secondary' in structure
        assert 'derived' in structure
    
    def test_structure_separates_types(self, hierarchy):
        """Test that structure separates objective types"""
        structure = hierarchy.get_hierarchy_structure()
        
        # Should have primary and secondary objectives
        assert len(structure['primary']) > 0
        assert len(structure['secondary']) > 0
    
    def test_structure_includes_relationships(self, hierarchy):
        """Test that structure includes relationships"""
        structure = hierarchy.get_hierarchy_structure()
        
        # Each objective should have dependencies and conflicts
        for obj_list in [structure['primary'], structure['secondary']]:
            for obj in obj_list:
                assert 'dependencies' in obj
                assert 'conflicts' in obj


class TestObjectiveQueries:
    """Test objective query methods"""
    
    def test_is_derived_objective(self, hierarchy):
        """Test checking if objective is derived"""
        # Add a derived objective
        obj = Objective(
            name='derived',
            description='Derived',
            objective_type=ObjectiveType.DERIVED
        )
        hierarchy.add_objective(obj)
        
        assert hierarchy.is_derived_objective('derived') is True
        assert hierarchy.is_derived_objective('prediction_accuracy') is False
    
    def test_is_derived_nonexistent(self, hierarchy):
        """Test is_derived on nonexistent objective"""
        assert hierarchy.is_derived_objective('nonexistent') is False
    
    def test_get_parents(self, hierarchy):
        """Test getting parent objectives"""
        # Add child objective
        obj = Objective(name='child', description='Child')
        hierarchy.add_objective(obj, parent='prediction_accuracy')
        
        parents = hierarchy.get_parents('child')
        
        assert 'prediction_accuracy' in parents
    
    def test_get_parents_nonexistent(self, hierarchy):
        """Test getting parents of nonexistent objective"""
        parents = hierarchy.get_parents('nonexistent')
        
        assert isinstance(parents, list)
        assert len(parents) == 0
    
    def test_get_children(self, hierarchy):
        """Test getting child objectives"""
        # Add child
        obj = Objective(name='child', description='Child')
        hierarchy.add_objective(obj, parent='prediction_accuracy')
        
        children = hierarchy.get_children('prediction_accuracy')
        
        assert 'child' in children
    
    def test_get_children_nonexistent(self, hierarchy):
        """Test getting children of nonexistent objective"""
        children = hierarchy.get_children('nonexistent')
        
        assert isinstance(children, list)
        assert len(children) == 0


class TestObjectiveValues:
    """Test objective value management"""
    
    def test_update_objective_value(self, hierarchy):
        """Test updating objective value"""
        hierarchy.update_objective_value('prediction_accuracy', 0.92)
        
        assert hierarchy.objectives['prediction_accuracy'].current_value == 0.92
    
    def test_update_nonexistent_objective(self, hierarchy):
        """Test updating nonexistent objective (should not crash)"""
        hierarchy.update_objective_value('nonexistent', 0.5)
        
        # Should handle gracefully
        assert True
    
    def test_get_unsatisfied_objectives(self, hierarchy):
        """Test getting unsatisfied objectives"""
        # Set some values
        hierarchy.update_objective_value('prediction_accuracy', 0.5)  # Far from target
        
        unsatisfied = hierarchy.get_unsatisfied_objectives()
        
        assert isinstance(unsatisfied, list)
        assert 'prediction_accuracy' in unsatisfied
    
    def test_get_violated_objectives(self, hierarchy):
        """Test getting violated objectives"""
        # Set value outside constraints
        hierarchy.update_objective_value('prediction_accuracy', 0.5)  # Below min of 0.9
        
        violated = hierarchy.get_violated_objectives()
        
        assert isinstance(violated, list)
        assert 'prediction_accuracy' in violated


class TestConflictMatrix:
    """Test conflict matrix computation"""
    
    def test_compute_conflict_matrix(self, hierarchy):
        """Test computing conflict matrix"""
        matrix = hierarchy.compute_conflict_matrix()
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.shape == (len(hierarchy.objectives), len(hierarchy.objectives))
    
    def test_conflict_matrix_diagonal_zero(self, hierarchy):
        """Test that diagonal is zero (no self-conflict)"""
        matrix = hierarchy.compute_conflict_matrix()
        
        n = matrix.shape[0]
        for i in range(n):
            assert matrix[i, i] == 0.0
    
    def test_conflict_matrix_cached(self, hierarchy):
        """Test that conflict matrix is cached"""
        matrix1 = hierarchy.compute_conflict_matrix()
        matrix2 = hierarchy.compute_conflict_matrix()
        
        assert np.array_equal(matrix1, matrix2)


class TestStatistics:
    """Test statistics"""
    
    def test_get_statistics(self, hierarchy):
        """Test getting statistics"""
        stats = hierarchy.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_objectives' in stats
        assert 'primary_objectives' in stats
        assert 'secondary_objectives' in stats
    
    def test_statistics_counts_correct(self, hierarchy):
        """Test that statistics counts are correct"""
        stats = hierarchy.get_statistics()
        
        total = (stats['primary_objectives'] + 
                stats['secondary_objectives'] + 
                stats['derived_objectives'])
        
        assert total == stats['total_objectives']
    
    def test_statistics_include_relationships(self, hierarchy):
        """Test that statistics include relationship counts"""
        stats = hierarchy.get_statistics()
        
        assert 'total_dependencies' in stats
        assert 'total_conflicts' in stats


class TestThreadSafety:
    """Test thread safety"""
    
    def test_concurrent_add_objectives(self, empty_hierarchy):
        """Test concurrent objective addition"""
        import threading
        
        errors = []
        
        def add_objective(i):
            try:
                obj = Objective(name=f'obj_{i}', description=f'Objective {i}')
                empty_hierarchy.add_objective(obj)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=add_objective, args=(i,)) 
                  for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(empty_hierarchy.objectives) >= 10
    
    def test_concurrent_queries(self, hierarchy):
        """Test concurrent queries are thread-safe"""
        import threading
        
        results = []
        errors = []
        
        def query():
            try:
                order = hierarchy.get_priority_order()
                deps = hierarchy.get_dependencies('prediction_accuracy')
                results.append((order, deps))
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=query) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10


class TestEdgeCases:
    """Test edge cases"""
    
    def test_empty_constraints(self):
        """Test objective with empty constraints"""
        obj = Objective(
            name='test',
            description='Test',
            constraints={},
            current_value=0.5
        )
        
        assert obj.violates_constraints() is False
    
    def test_self_dependency(self, empty_hierarchy):
        """Test handling of self-dependency"""
        obj = Objective(
            name='self_dep',
            description='Self-dependent',
            dependencies=['self_dep']
        )
        
        empty_hierarchy.add_objective(obj)
        
        # Should be caught in consistency check
        result = empty_hierarchy.check_consistency()
        assert result['consistent'] is False
    
    def test_very_long_dependency_chain(self, empty_hierarchy):
        """Test handling of very long dependency chains"""
        # Create chain of 20 objectives
        for i in range(20):
            deps = [f'obj_{i-1}'] if i > 0 else []
            obj = Objective(
                name=f'obj_{i}',
                description=f'Objective {i}',
                dependencies=deps
            )
            empty_hierarchy.add_objective(obj)
        
        # Should handle gracefully
        deps = empty_hierarchy.get_transitive_dependencies('obj_19')
        assert len(deps) == 19
    
    def test_multiple_parents(self, empty_hierarchy):
        """Test objective with multiple implicit parents"""
        obj_a = Objective(name='a', description='A')
        obj_b = Objective(name='b', description='B')
        obj_c = Objective(name='c', description='C', dependencies=['a', 'b'])
        
        empty_hierarchy.add_objective(obj_a)
        empty_hierarchy.add_objective(obj_b)
        empty_hierarchy.add_objective(obj_c)
        
        # Should handle multiple dependencies
        deps = empty_hierarchy.get_dependencies('c')
        assert len(deps) == 2


class TestIntegration:
    """Integration tests"""
    
    # --- FIX 3: (Correct) ---
    # Use the pristine_hierarchy fixture
    def test_full_workflow(self, pristine_hierarchy):
    # --- END FIX 3 ---
        """Test full workflow of adding and managing objectives"""
        # 1. Add objectives
        obj_a = Objective(name='a', description='A', priority=0)
        obj_b = Objective(name='b', description='B', priority=1, dependencies=['a'])
        
        # Use the pristine, empty hierarchy
        pristine_hierarchy.add_objective(obj_a)
        pristine_hierarchy.add_objective(obj_b)
        
        # 2. Check consistency
        result = pristine_hierarchy.check_consistency()
        assert result['consistent'] is True
        
        # 3. Get priority order
        order = pristine_hierarchy.get_priority_order()
        assert order[0] == 'a'  # Higher priority first
        
        # 4. Update values
        pristine_hierarchy.update_objective_value('a', 0.95)
        
        # 5. Get statistics
        stats = pristine_hierarchy.get_statistics()
        # This will now be 2, not 6
        assert stats['total_objectives'] == 2
    
    # --- FIX 3: (Correct) ---
    # Use the pristine_hierarchy fixture
    def test_conflict_detection_workflow(self, pristine_hierarchy):
    # --- END FIX 3 ---
        """Test workflow for conflict detection"""
        # 1. Add conflicting objectives
        obj_a = Objective(
            name='speed',
            description='Maximize speed',
            conflicts_with=['accuracy']
        )
        obj_b = Objective(
            name='accuracy',
            description='Maximize accuracy'
        )
        
        # Use the pristine, empty hierarchy
        pristine_hierarchy.add_objective(obj_a)
        pristine_hierarchy.add_objective(obj_b)
        
        # 2. Find conflicts
        conflict = pristine_hierarchy.find_conflicts('speed', 'accuracy')
        assert conflict is not None
        
        # 3. Compute conflict matrix
        matrix = pristine_hierarchy.compute_conflict_matrix()
        # This will now be (2, 2), not (6, 6)
        assert matrix.shape == (2, 2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])