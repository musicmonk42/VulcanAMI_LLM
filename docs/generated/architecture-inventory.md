# Generated architecture inventory

Digest: `bf2bb3e865b2a37f186025e3dff7f12a699d5c86cf197a4bf151eb82e0154d7c`

This file is generated from `docs/generated/architecture-inventory.json`.

## Entrypoints

Count: 404

- `src/vulcan/tests/test_multimodal_reasoning.py:999` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_curiosity_engine_integration.py:1000` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_curiosity_engine_core.py:1002` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_agent_lifecycle.py:1005` kind='python_main_guard', reachability='test-only'
- `tests/test_graph_validation.py:1008` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_task_queues.py:1008` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_probabilistic_reasoning.py:1017` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_metrics.py:1016` kind='python_main_guard', reachability='test-only'
- `src/consensus_manager.py:953` kind='python_main_guard', reachability='unknown'
- `src/vulcan/tests/test_distributed.py:1024` kind='python_main_guard', reachability='test-only'
- `src/governance_loop.py:938` kind='python_main_guard', reachability='unknown'
- `src/generate_transparency_report.py:1023` kind='python_main_guard', reachability='unknown'
- `src/vulcan/tests/test_extracted_modules.py:1035` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_symbolic_core.py:1040` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_knowledge_crystallizer_core.py:1045` kind='python_main_guard', reachability='test-only'
- `src/run_validation_test.py:999` kind='python_main_guard', reachability='unknown'
- `tests/test_ai_runtime_integration.py:1047` kind='python_main_guard', reachability='test-only'
- `tests/test_runtime_extensions.py:1048` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_problem_executor.py:1053` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_learning_module_intergration.py:1019` kind='python_main_guard', reachability='test-only'
- `tests/test_registry_api_server.py:1061` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_adversarial_formal.py:1063` kind='python_main_guard', reachability='test-only'
- `src/training/train_llm_with_self_improvement.py:1063` kind='python_main_guard', reachability='unknown'
- `src/vulcan/tests/test_compliance_bias.py:1070` kind='python_main_guard', reachability='test-only'
- `src/consensus_engine.py:958` kind='python_main_guard', reachability='unknown'
- `src/vulcan/tests/test_symbolic_advanced.py:1072` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_memory_integration.py:1073` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_safety_types.py:1084` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_world_model_router.py:1086` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_main.py:1089` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_analogical_reasoning.py:1091` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_conflict_resolver.py:1094` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_intervention_manager.py:1097` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_gap_analyzer.py:1098` kind='python_main_guard', reachability='test-only'
- `src/data_augmentor.py:1007` kind='python_main_guard', reachability='unknown'
- `src/vulcan/tests/test_retrieval.py:1101` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_principle_extractor.py:1108` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_rollback_audit.py:1108` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_vulcan_cost_model.py:1108` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_domain_validators.py:1119` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_hierarchical.py:1122` kind='python_main_guard', reachability='test-only'
- `tests/test_os_import_unified_chat.py:69` kind='python_main_guard', reachability='test-only'
- `src/audit_log.py:1074` kind='python_main_guard', reachability='unknown'
- `tests/test_ontology_validation.py:1132` kind='python_main_guard', reachability='test-only'
- `tests/test_demo_graphix.py:1145` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_contraindication_tracker.py:1157` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_exploration_budget.py:1158` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_principle_learner.py:1163` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_dynamics_model.py:1169` kind='python_main_guard', reachability='test-only'
- `src/vulcan/tests/test_prediction_engine.py:1172` kind='python_main_guard', reachability='test-only'
- ... 354 more in JSON

## Routes

Count: 425

- `src/platform/routes_admin.py:104` handler='admin_get_service', method='GET', reachability='unknown', route='/admin/services/{service_name}'
- `src/full_platform.py:3742` handler='admin_get_service', method='GET', reachability='unknown', route='/admin/services/{service_name}'
- `src/platform/routes_admin.py:70` handler='admin_list_services', method='GET', reachability='unknown', route='/admin/services'
- `src/full_platform.py:3711` handler='admin_list_services', method='GET', reachability='unknown', route='/admin/services'
- `src/platform/routes_admin.py:187` handler='admin_start_service', method='POST', reachability='unknown', route='/admin/services/{service_name}/start'
- `src/full_platform.py:3818` handler='admin_start_service', method='POST', reachability='unknown', route='/admin/services/{service_name}/start'
- `src/platform/routes_admin.py:143` handler='admin_stop_service', method='POST', reachability='unknown', route='/admin/services/{service_name}/stop'
- `src/full_platform.py:3778` handler='admin_stop_service', method='POST', reachability='unknown', route='/admin/services/{service_name}/stop'
- `src/platform/routes_adversarial.py:268` handler='adversarial_status', method='GET', reachability='unknown', route='/api/adversarial/status'
- `src/full_platform.py:4947` handler='adversarial_status', method='GET', reachability='unknown', route='/api/adversarial/status'
- `src/vulcan/runtime/app.py:188` handler='alignment', method='POST', reachability='production-reachable', route='/v1/admin/alignment'
- `src/platform/routes_health_ext.py:118` handler='api_status', method='GET', reachability='unknown', route='/api/status'
- `src/full_platform.py:3683` handler='api_status', method='GET', reachability='unknown', route='/api/status'
- `src/vulcan/endpoints/self_improvement.py:295` handler='approve_improvement', method='POST', reachability='unknown', route='/v1/improvement/approve'
- `src/full_platform.py:4637` handler='arena_feedback', method='POST', reachability='unknown', route='/api/arena/feedback'
- `src/platform/routes_arena.py:88` handler='arena_feedback', method='POST', reachability='unknown', route='/api/arena/feedback'
- `src/platform/routes_arena.py:120` handler='arena_feedback_dispatch', method='POST', reachability='unknown', route='/api/arena/feedback_dispatch'
- `src/full_platform.py:4669` handler='arena_feedback_dispatch', method='POST', reachability='unknown', route='/api/arena/feedback_dispatch'
- `src/graphix_arena.py:3686` handler='arena_health', method='GET', reachability='unknown', route='/health'
- `src/full_platform.py:4620` handler='arena_run_agent', method='POST', reachability='unknown', route='/api/arena/run/{agent_id}'
- `src/platform/routes_arena.py:71` handler='arena_run_agent', method='POST', reachability='unknown', route='/api/arena/run/{agent_id}'
- `src/platform/routes_arena.py:103` handler='arena_tournament', method='POST', reachability='unknown', route='/api/arena/tournament'
- `src/full_platform.py:4652` handler='arena_tournament', method='POST', reachability='unknown', route='/api/arena/tournament'
- `src/vulcan/runtime/app.py:193` handler='audit_case', method='GET', reachability='production-reachable', route='/v1/audit/cases/{case_id}'
- `src/vulcan/runtime/app.py:240` handler='audit_improvement', method='GET', reachability='production-reachable', route='/v1/audit/improvements/{proposal_digest}'
- `src/vulcan/runtime/app.py:173` handler='capabilities', method='GET', reachability='production-reachable', route='/v1/capabilities'
- `src/platform/routes_adversarial.py:324` handler='check_query_adversarial', method='POST', reachability='unknown', route='/api/adversarial/check-query'
- `src/full_platform.py:5003` handler='check_query_adversarial', method='POST', reachability='unknown', route='/api/adversarial/check-query'
- `src/vulcan/endpoints/distillation.py:279` handler='clear_distillation_buffer', method='POST', reachability='unknown', route='/clear'
- `src/vulcan/endpoints/config.py:332` handler='clear_distillation_buffer', method='DELETE', reachability='unknown', route='/v1/distillation/buffer'
- `src/vulcan/endpoints/status.py:107` handler='cognitive_status', method='GET', reachability='unknown', route='/v1/cognitive/status'
- `src/platform/routes_health_ext.py:18` handler='component_health', method='GET', reachability='unknown', route='/health/components'
- `src/full_platform.py:3587` handler='component_health', method='GET', reachability='unknown', route='/health/components'
- `src/api_gateway.py:310` handler='component_health', method='GET', reachability='unknown', route='/health/components'
- `src/vulcan/endpoints/planning.py:32` handler='create_plan', method='POST', reachability='unknown', route='/v1/plan'
- `src/vulcan/endpoints/status.py:510` handler='debug_deployment', method='GET', reachability='unknown', route='/debug/deployment'
- `src/platform/routes_vulcan.py:124` handler='debug_parent_deployment', method='GET', reachability='unknown', route='/debug/deployment'
- `src/full_platform.py:4048` handler='debug_parent_deployment', method='GET', reachability='unknown', route='/debug/deployment'
- `src/governance/app.py:498` handler='deploy_proposal', method='ROUTE', reachability='unknown', route='/proposals/<proposal_id>/deploy'
- `src/pii_service.py:391` handler='detect_pii', method='POST', reachability='unknown', route='/detect'
- `src/vulcan/runtime/app.py:183` handler='domains', method='POST', reachability='production-reachable', route='/v1/admin/domains'
- `src/vulcan/endpoints/world_model.py:84` handler='execute_causal_intervention', method='POST', reachability='unknown', route='/intervene'
- `src/vulcan/endpoints/execution.py:30` handler='execute_step', method='POST', reachability='unknown', route='/v1/step'
- `src/vulcan/endpoints/reasoning.py:97` handler='explain', method='POST', reachability='unknown', route='/llm/explain'
- `src/graphix_arena.py:3651` handler='feedback_dispatch_info', method='GET', reachability='unknown', route='/api/feedback_dispatch'
- `src/graphix_arena.py:3606` handler='feedback_endpoint', method='POST', reachability='unknown', route='/api/feedback_dispatch'
- `src/vulcan/endpoints/world_model.py:139` handler='generate_counterfactual_prediction', method='POST', reachability='unknown', route='/predict'
- `src/governance/app.py:636` handler='get_agent', method='ROUTE', reachability='unknown', route='/agents/<agent_id>'
- `src/vulcan/endpoints/agents.py:48` handler='get_agent_pool_status', method='GET', reachability='unknown', route='/status'
- `src/governance/app.py:577` handler='get_audit_log', method='ROUTE', reachability='unknown', route='/audit/log'
- ... 375 more in JSON

## Import Identities

Count: 1693

- `tests/security/test_output_firewall.py:1` module='vulcan.runtime.output', reachability='test-only'
- `tests/test_probabilistic_routing.py:10` module='src.vulcan.reasoning.probabilistic_reasoning', reachability='test-only'
- `tests/test_wm_request_handling.py:10` module='src.vulcan.world_model.request_handling', reachability='test-only'
- `tests/assurance/test_capability_registry.py:10` module='vulcan.assurance.capabilities', reachability='test-only'
- `tests/core/test_core_primitives.py:10` module='vulcan.core.decisions', reachability='test-only'
- `src/vulcan/endpoints/__init__.py:10` module='vulcan.endpoints.distillation', reachability='unknown'
- `tests/test_progressive_research.py:10` module='vulcan.learning_owner', reachability='test-only'
- `tests/test_world_model_research.py:10` module='vulcan.learning_owner', reachability='test-only'
- `tests/security/test_local_language_governance.py:10` module='vulcan.local_language.tokenizer', reachability='test-only'
- `src/vulcan/tests/test_consolidation.py:10` module='vulcan.memory.base', reachability='test-only'
- `src/vulcan/runtime/kernel.py:10` module='vulcan.memory.governed', reachability='production-reachable'
- `src/vulcan/runtime/audit.py:10` module='vulcan.persistence.audit.reconcile', reachability='production-reachable'
- `tests/test_learning_outbox.py:10` module='vulcan.runtime.audit', reachability='test-only'
- `tests/runtime/test_production_composition.py:10` module='vulcan.runtime.composition', reachability='test-only'
- `tests/security/test_persistent_audit_alignment.py:10` module='vulcan.runtime.finalization', reachability='test-only'
- `tests/security/test_real_response_safety_composition.py:10` module='vulcan.runtime.kernel', reachability='test-only'
- `tests/security/test_runtime_convergence.py:10` module='vulcan.runtime.kernel', reachability='test-only'
- `tests/security/test_language_contracts.py:10` module='vulcan.runtime.semantic', reachability='test-only'
- `src/vulcan/runtime/self_improvement.py:10` module='vulcan.world_model.meta_reasoning.csiu_enforcement', reachability='production-reachable'
- `tests/test_phase9g_runtime_ownership.py:10` module='vulcan.world_model.meta_reasoning.self_improvement_drive', reachability='test-only'
- `src/vulcan/monitoring/__init__.py:3` module='vulcan.monitoring.memory_guard', reachability='unknown'
- `src/vulcan/tests/test_meta_learning.py:5` module='vulcan.learning.meta_learning', reachability='test-only'
- `src/vulcan/tests/test_continual_learning.py:6` module='vulcan.learning.continual_learning', reachability='test-only'
- `tests/test_phase9d_governed_drive_e2e.py:6` module='vulcan.world_model.meta_reasoning.governed_transaction', reachability='test-only'
- `src/vulcan/processing.py:100` module='src.vulcan.simple_mode', reachability='unknown'
- `tests/test_philosophical_reasoning_registration.py:100` module='vulcan.reasoning.reasoning_types', reachability='test-only'
- `src/vulcan/reasoning/analogical/base_reasoner.py:1003` module='vulcan.reasoning.analogical.types', reachability='unknown'
- `src/vulcan/world_model/world_model_core.py:1003` module='vulcan.world_model.knowledge_handler', reachability='unknown'
- `src/vulcan/api_gateway.py:101` module='src.vulcan.learning', reachability='unknown'
- `src/vulcan/endpoints/planning.py:101` module='vulcan.reasoning.singletons', reachability='unknown'
- `tests/test_conclusion_loss_fixes.py:101` module='vulcan.reasoning.unified.strategies', reachability='test-only'
- `src/vulcan/tests/test_vulcan_cost_model.py:1010` module='vulcan.reasoning.selection.cost_model', reachability='test-only'
- `src/vulcan/learning/__init__.py:1012` module='vulcan.reasoning.selection.semantic_tool_matcher', reachability='unknown'
- `src/vulcan/tests/test_extracted_modules.py:1014` module='vulcan.utils_main.timing', reachability='test-only'
- `src/vulcan/distillation/__init__.py:102` module='vulcan.distillation.promotion_gate', reachability='unknown'
- `src/vulcan/tests/test_unified_chat_imports.py:102` module='vulcan.endpoints', reachability='test-only'
- `src/vulcan/tests/test_planning_endpoint.py:102` module='vulcan.endpoints.planning', reachability='test-only'
- `tests/security/test_runtime_convergence.py:102` module='vulcan.runtime.app', reachability='test-only'
- `src/vulcan/llm/__init__.py:89` module='vulcan.llm.openai_client', reachability='unknown'
- `src/vulcan/tests/test_metrics_centralization.py:93` module='vulcan.metrics', reachability='test-only'
- `src/vulcan/orchestrator/platform_status.py:99` module='vulcan.arena', reachability='unknown'
- `src/vulcan/world_model/world_model_core.py:1020` module='vulcan.world_model.creative_handler', reachability='unknown'
- `src/api_server.py:103` module='vulcan.reasoning.unified', reachability='unknown'
- `src/vulcan/world_model/world_model_core.py:1037` module='vulcan.world_model.llm_guidance', reachability='unknown'
- `src/vulcan/server/startup/manager.py:1039` module='vulcan.utils_main.http_session', reachability='unknown'
- `src/vulcan/routing/telemetry_recorder.py:104` module='src.vulcan.memory.hierarchical', reachability='unknown'
- `src/vulcan/api_gateway.py:104` module='vulcan.planning', reachability='unknown'
- `src/api_server.py:104` module='vulcan.reasoning.reasoning_types', reachability='unknown'
- `src/vulcan/utils_main/__init__.py:104` module='vulcan.utils_main.network', reachability='unknown'
- `src/vulcan/reasoning/singletons.py:1042` module='vulcan.reasoning.multimodal_reasoning', reachability='unknown'
- ... 1643 more in JSON

## Singletons

Count: 16

- `src/unified_runtime/graph_validator.py:1097` name='get_global_validator', reachability='unknown'
- `src/vulcan/reasoning/singletons.py:1193` name='get_singleton', reachability='unknown'
- `src/vulcan/problem_decomposer/decomposer_bootstrap.py:57` name='create_strategy_instances', reachability='unknown'
- `src/unified_runtime/execution_engine.py:1483` name='get_global_engine', reachability='unknown'
- `src/integration/parallel_candidate_scorer.py:1704` name='get_global_scorer', reachability='unknown'
- `src/vulcan/processing.py:227` name='get_instance', reachability='unknown'
- `src/vulcan/schema_registry.py:270` name='get_instance', reachability='unknown'
- `scripts/bootstrap_knowledge.py:278` name='get_instance', reachability='research-only'
- `src/vulcan/orchestrator/agent_pool.py:378` name='get_instance', reachability='unknown'
- `src/vulcan/orchestrator/agent_pool.py:450` name='get_all_instances', reachability='unknown'
- `src/full_platform.py:4576` name='get_arena_instance', reachability='unknown'
- `src/vulcan/reasoning/selection/memory_prior.py:34` name='get_global_embedding_model', reachability='unknown'
- `src/platform/routes_arena.py:27` name='get_arena_instance', reachability='unknown'
- `tests/test_nso_aligner_singleton_lifecycle.py:74` name='create_instance', reachability='test-only'
- `tests/test_nso_aligner.py:923` name='get_instance', reachability='test-only'
- `src/vulcan/safety/tool_safety.py:905` name='get_global_safety_stats', reachability='unknown'

## Workers

Count: 453

- `src/vulcan/server/startup/manager.py:657` call='Thread', reachability='unknown'
- `src/vulcan/learning/rlhf_feedback.py:1040` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/safety/neural_safety.py:1052` call='asyncio.create_task', reachability='unknown'
- `src/platform/lifespan.py:109` call='asyncio.create_task', reachability='unknown'
- `tests/test_scheduler_node.py:115` call='asyncio.create_task', reachability='test-only'
- `src/unified_runtime/unified_runtime_core.py:1163` call='asyncio.create_task', reachability='unknown'
- `tests/test_scheduler_node.py:129` call='asyncio.create_task', reachability='test-only'
- `src/vulcan/api_gateway.py:1420` call='asyncio.create_task', reachability='unknown'
- `tests/test_scheduler_node.py:148` call='asyncio.create_task', reachability='test-only'
- `src/graphix_arena.py:1595` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/api_gateway.py:1724` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/api_gateway.py:1725` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/api_gateway.py:1732` call='asyncio.create_task', reachability='unknown'
- `src/full_platform.py:2148` call='asyncio.create_task', reachability='unknown'
- `src/evolve/self_optimizer.py:222` call='asyncio.create_task', reachability='unknown'
- `src/full_platform.py:2233` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/llm/hybrid_executor.py:2241` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/llm/hybrid_executor.py:2245` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/safety/safety_validator.py:2479` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/safety/safety_validator.py:2490` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/safety/safety_validator.py:2501` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/safety/safety_validator.py:2513` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/safety/safety_validator.py:2521` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/safety/safety_validator.py:2532` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/api_gateway.py:262` call='asyncio.create_task', reachability='unknown'
- `src/full_platform.py:2891` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/processing.py:2957` call='asyncio.create_task', reachability='unknown'
- `src/scheduler_node.py:297` call='asyncio.create_task', reachability='unknown'
- `src/full_platform.py:2990` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/orchestrator/variants.py:315` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/orchestrator/variants.py:318` call='asyncio.create_task', reachability='unknown'
- `src/scheduler_node.py:371` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/api_gateway.py:383` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/arena/client.py:419` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/orchestrator/variants.py:509` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/orchestrator/variants.py:512` call='asyncio.create_task', reachability='unknown'
- `tests/test_scheduler_node.py:57` call='asyncio.create_task', reachability='test-only'
- `src/vulcan/curiosity_engine/curiosity_driver.py:639` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/curiosity_engine/outcome_bridge.py:642` call='asyncio.create_task', reachability='unknown'
- `tests/test_graphix_client.py:651` call='asyncio.create_task', reachability='test-only'
- `src/unified_runtime/execution_engine.py:688` call='asyncio.create_task', reachability='unknown'
- `src/integration/token_consensus_adapter.py:71` call='asyncio.create_task', reachability='unknown'
- `tests/test_scheduler_node.py:73` call='asyncio.create_task', reachability='test-only'
- `src/audit_log.py:746` call='asyncio.create_task', reachability='unknown'
- `tests/test_scheduler_node.py:76` call='asyncio.create_task', reachability='test-only'
- `scripts/load_test.py:84` call='asyncio.create_task', reachability='research-only'
- `src/vulcan/safety/neural_safety.py:919` call='asyncio.create_task', reachability='unknown'
- `tests/test_scheduler_node.py:92` call='asyncio.create_task', reachability='test-only'
- `src/unified_runtime/execution_engine.py:920` call='asyncio.create_task', reachability='unknown'
- `src/vulcan/learning/rlhf_feedback.py:955` call='asyncio.create_task', reachability='unknown'
- ... 403 more in JSON

## Fallbacks

Count: 4427

- `src/vulcan/endpoints/self_improvement.py:98` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/safety/rollback_audit.py:99` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/memory/governed_unlearning.py:998` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/curiosity_engine/curiosity_engine_core.py:999` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/learning/metacognition.py:1000` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/tests/test_tool_selector.py:1000` kind='except_exception', reachability='test-only', review='unreviewed'
- `src/vulcan/reasoning/mathematical_verification.py:994` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/endpoints/unified_chat.py:998` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/tests/test_retrieval.py:1000` kind='except_exception', reachability='test-only', review='unreviewed'
- `src/vulcan/utils/safe_execution.py:979` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/world_model/world_model_router.py:1000` kind='except_exception', reachability='unknown', review='unreviewed'
- `tests/performance/scalability_stress_test.py:1000` kind='except_exception', reachability='test-only', review='unreviewed'
- `tests/test_registry_api_server.py:1000` kind='except_exception', reachability='test-only', review='unreviewed'
- `src/nso_aligner.py:1002` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/safety/neural_safety.py:1002` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/server/startup/manager.py:1002` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/reasoning/singletons.py:1002` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/tests/test_collective.py:1003` kind='except_exception', reachability='test-only', review='unreviewed'
- `src/vulcan/curiosity_engine/experiment_generator.py:1003` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/world_model/system_observer.py:1004` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/training/governed_trainer.py:1004` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/world_model/world_model_core.py:1006` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/reasoning/selection/warm_pool.py:1006` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/unified_runtime/node_handlers.py:1007` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/learning/__init__.py:1007` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/reasoning/multimodal_reasoning.py:1006` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/safety/safety_status_endpoint.py:97` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/world_model/self_improvement_engine.py:97` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/governance_loop.py:99` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/server/app.py:99` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/reasoning/unified/orchestrator.py:1008` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/orchestrator/task_queues.py:1009` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/reasoning/causal_reasoning.py:1009` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/learning/continual_learning.py:1010` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/reasoning/selection/portfolio_executor.py:1010` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/generate_transparency_report.py:1011` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/semantic_bridge/semantic_bridge_core.py:1011` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/governance_loop.py:1012` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/integration/graphix_vulcan_bridge.py:1013` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/curiosity_engine/exploration_budget.py:1015` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/reasoning/selection/memory_prior.py:1015` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/curiosity_engine/dependency_graph.py:1016` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/orchestrator/task_queues.py:1017` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/reasoning/analogical/structure_mapping.py:1017` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/generate_transparency_report.py:1018` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/world_model/state_save.py:101` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/generate_transparency_report.py:1014` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/knowledge_crystallizer/principle_extractor.py:1018` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/vulcan/curiosity_engine/curiosity_engine_core.py:1012` kind='except_exception', reachability='unknown', review='unreviewed'
- `src/memory/governed_unlearning.py:1019` kind='except_exception', reachability='unknown', review='unreviewed'
- ... 4377 more in JSON

## Mocks

Count: 400

- `tests/test_distillation_webhook_async.py:100` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_llm_router_autodiscovery.py:102` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_audit_log.py:1020` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_audit_log.py:1022` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_audit_log.py:1023` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_defect_fixes.py:103` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_survival_protocol.py:103` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_runtime_extensions.py:1034` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_registry_api_server.py:1053` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_faiss_config.py:106` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_survival_protocol.py:107` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_faiss_config.py:109` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_safety_status_endpoint.py:118` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_safety_status_endpoint.py:122` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_tool_selector.py:1260` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_nso_aligner_singleton_lifecycle.py:127` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_llm_router_autodiscovery.py:128` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_tool_selector.py:1288` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_tool_selector.py:1313` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_tool_selector.py:1332` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_tool_selector.py:1360` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_tool_selector.py:1380` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_tool_selector.py:1395` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_faiss_config.py:143` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_setup_agent.py:147` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_safety_status_endpoint.py:149` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/llm/mock_llm.py:151` kind='MagicMock', reachability='unknown', review='unreviewed'
- `src/vulcan/tests/test_safety_status_endpoint.py:151` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_run_validation_test.py:153` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_vulcan_reasoning_output.py:153` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_safety_status_endpoint.py:154` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_llm_router_autodiscovery.py:154` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_vulcan_reasoning_output.py:154` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_llm_validators.py:155` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_reasoning_content_propagation.py:155` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_run_validation_test.py:155` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_llm_validators.py:160` kind='MagicMock', reachability='test-only', review='unreviewed'
- `tests/test_vulcan_reasoning_output.py:161` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_llm_validators.py:166` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_llm_validators.py:167` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_dqs_integration.py:168` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/world_model/meta_reasoning/transparency_interface.py:169` kind='MagicMock', reachability='unknown', review='unreviewed'
- `src/vulcan/tests/test_cli_interactive.py:170` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_cli_interactive.py:171` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_cli_interactive.py:172` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_llm_validators.py:172` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/world_model/meta_reasoning/counterfactual_objectives.py:172` kind='MagicMock', reachability='unknown', review='unreviewed'
- `tests/test_unified_runtime_core.py:171` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/tests/test_cli_interactive.py:173` kind='MagicMock', reachability='test-only', review='unreviewed'
- `src/vulcan/world_model/meta_reasoning/transparency_interface.py:173` kind='MagicMock', reachability='unknown', review='unreviewed'
- ... 350 more in JSON

## Persistence

Count: 334

- `tests/persistence/test_memory_outbox_v2.py:103` kind='call', reachability='test-only', value='SQLiteMemoryRepository'
- `src/vulcan/runtime/alignment.py:107` kind='call', reachability='production-reachable', value='fcntl.flock'
- `src/platform/background.py:108` kind='call', reachability='unknown', value='fcntl.flock'
- `tests/persistence/test_memory_outbox_v2.py:108` kind='call', reachability='test-only', value='SQLiteMemoryRepository'
- `src/adversarial_tester.py:1109` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/vulcan/learning/outbox.py:111` kind='call', reachability='unknown', value='fcntl.flock'
- `src/vulcan/safety/adversarial_integration.py:1166` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/vulcan/safety/rollback_audit.py:1230` kind='call', reachability='unknown', value='sqlite3.connect'
- `tests/test_learning_outbox.py:125` kind='call', reachability='test-only', value='sqlite3.connect'
- `tests/security/test_governed_memory.py:13` kind='call', reachability='test-only', value='SQLiteMemoryRepository'
- `src/vulcan/knowledge_crystallizer/knowledge_storage.py:1301` kind='call', reachability='unknown', value='self._load_from_sqlite'
- `src/vulcan/knowledge_crystallizer/knowledge_storage.py:1311` kind='call', reachability='unknown', value='self._init_sqlite'
- `src/vulcan/knowledge_crystallizer/knowledge_storage.py:1315` kind='call', reachability='unknown', value='self._init_sqlite'
- `src/vulcan/knowledge_crystallizer/knowledge_storage.py:1325` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/vulcan/runtime/audit.py:134` kind='call', reachability='production-reachable', value='fcntl.flock'
- `src/listener.py:138` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/adversarial_tester.py:158` kind='call', reachability='unknown', value='sqlite3.connect'
- `tests/security/test_governed_memory_architecture.py:17` kind='call', reachability='test-only', value='SQLiteMemoryRepository'
- `tests/security/test_governed_memory_architecture.py:19` kind='call', reachability='test-only', value='SQLiteMemoryRepository'
- `src/generate_transparency_report.py:192` kind='call', reachability='unknown', value='fcntl.flock'
- `src/vulcan/memory/governed.py:204` kind='call', reachability='unknown', value='fcntl.flock'
- `src/vulcan/memory/governed.py:208` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/generate_transparency_report.py:211` kind='call', reachability='unknown', value='fcntl.flock'
- `src/vulcan/memory/governed.py:213` kind='call', reachability='unknown', value='fcntl.flock'
- `src/persistence.py:211` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/vulcan/world_model/meta_reasoning/csiu_enforcement.py:225` kind='call', reachability='unknown', value='fcntl.flock'
- `src/security_audit_engine.py:233` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/ai_providers.py:239` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/vulcan/world_model/meta_reasoning/governed_transaction.py:244` kind='call', reachability='unknown', value='fcntl.flock'
- `src/vulcan/safety/governance_alignment.py:248` kind='call', reachability='unknown', value='sqlite3.connect'
- `tests/performance/performance_monitor.py:257` kind='call', reachability='test-only', value='self._get_sqlite_size'
- `tests/performance/performance_monitor.py:258` kind='call', reachability='test-only', value='self._get_sqlite_wal_size'
- `src/vulcan/safety/rollback_audit.py:256` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/vulcan/world_model/meta_reasoning/csiu_enforcement.py:270` kind='call', reachability='unknown', value='fcntl.flock'
- `src/full_platform.py:279` kind='call', reachability='unknown', value='fcntl.flock'
- `src/vulcan/world_model/meta_reasoning/governed_transaction.py:290` kind='call', reachability='unknown', value='fcntl.flock'
- `src/vulcan/world_model/meta_reasoning/governed_transaction.py:302` kind='call', reachability='unknown', value='fcntl.flock'
- `src/vulcan/curiosity_engine/resolution_bridge.py:313` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/full_platform.py:317` kind='call', reachability='unknown', value='fcntl.flock'
- `src/vulcan/world_model/meta_reasoning/governed_transaction.py:317` kind='call', reachability='unknown', value='fcntl.flock'
- `src/agent_registry.py:313` kind='call', reachability='unknown', value='sqlite3.connect'
- `tests/memory/test_memory_lifecycle_v3.py:33` kind='call', reachability='test-only', value='SQLiteMemoryRepository'
- `src/vulcan/world_model/meta_reasoning/governed_transaction.py:331` kind='call', reachability='unknown', value='fcntl.flock'
- `src/governance/registry_api_server.py:344` kind='call', reachability='unknown', value='sqlite3.connect'
- `tests/persistence/test_memory_outbox_v2.py:37` kind='call', reachability='test-only', value='sqlite3.connect'
- `tests/test_adversarial_tester.py:379` kind='call', reachability='test-only', value='sqlite3.connect'
- `src/listener.py:390` kind='call', reachability='unknown', value='sqlite3.connect'
- `src/api_server.py:396` kind='call', reachability='unknown', value='sqlite3.connect'
- `tests/persistence/test_memory_outbox_v2.py:44` kind='call', reachability='test-only', value='SQLiteMemoryRepository'
- `src/vulcan/memory/governed.py:444` kind='call', reachability='unknown', value='fcntl.flock'
- ... 284 more in JSON

## Environment Readers

Count: 431

- `src/audit_log.py:416` call='os.environ.get', key='AGENT_ID', reachability='unknown'
- `src/api_server.py:1058` call='os.environ.get', key='ALLOWED_ORIGIN', reachability='unknown'
- `src/api_server.py:143` call='os.environ.get', key='ALLOW_EPHEMERAL_SECRET', reachability='unknown'
- `src/governance/app.py:229` call='os.environ.get', key='ALLOW_LEGACY_AUTH', reachability='unknown'
- `src/ai_providers.py:1101` call='os.environ.get', key='ANTHROPIC_API_KEY', reachability='unknown'
- `src/api_gateway.py:662` call='os.environ.get', key='API_HOST', reachability='unknown'
- `src/api_gateway.py:663` call='os.environ.get', key='API_PORT', reachability='unknown'
- `src/audit_log.py:419` call='os.environ.get', key='APP_VERSION', reachability='unknown'
- `src/audit_log.py:135` call='os.environ.get', key='ASYNC_LOGGING', reachability='unknown'
- `src/audit_log.py:112` call='os.environ.get', key='AUDIT_LOG_BATCH_SIZE', reachability='unknown'
- `src/audit_log.py:115` call='os.environ.get', key='AUDIT_LOG_BATCH_TIMEOUT', reachability='unknown'
- `src/audit_log.py:99` call='os.environ.get', key='AUDIT_LOG_COMPRESSION', reachability='unknown'
- `src/audit_log.py:105` call='os.environ.get', key='AUDIT_LOG_ENCRYPT', reachability='unknown'
- `src/audit_log.py:109` call='os.environ.get', key='AUDIT_LOG_ENCRYPTION_KEY', reachability='unknown'
- `src/audit_log.py:88` call='os.environ.get', key='AUDIT_LOG_INTERVAL', reachability='unknown'
- `src/audit_log.py:92` call='os.environ.get', key='AUDIT_LOG_MAX_SIZE', reachability='unknown'
- `src/audit_log.py:79` call='os.environ.get', key='AUDIT_LOG_PATH', reachability='unknown'
- `src/audit_log.py:96` call='os.environ.get', key='AUDIT_LOG_RETENTION', reachability='unknown'
- `src/audit_log.py:83` call='os.environ.get', key='AUDIT_LOG_ROTATION', reachability='unknown'
- `src/api_server.py:194` call='os.environ.get', key='CALLBACK_DOMAIN_ALLOWLIST', reachability='unknown'
- `src/api_server.py:2080` call='os.environ.get', key='CA_CERT_PATH', reachability='unknown'
- `src/api_server.py:2078` call='os.environ.get', key='CERT_PATH', reachability='unknown'
- `src/api_server.py:2441` call='os.environ.get', key='CERT_PATH', reachability='unknown'
- `src/vulcan/tests/test_principle_extractor.py:1023` call='os.environ.get', key='CI', reachability='test-only'
- `src/vulcan/tests/test_principle_extractor.py:1024` call='os.environ.get', key='CI', reachability='test-only'
- `src/vulcan/tests/test_principle_extractor.py:1083` call='os.environ.get', key='CI', reachability='test-only'
- `src/vulcan/tests/test_principle_extractor.py:1084` call='os.environ.get', key='CI', reachability='test-only'
- `src/vulcan/memory/hierarchical.py:1656` call='os.environ.get', key='CI', reachability='unknown'
- `src/__init__.py:24` call='os.environ.get', key='CI', reachability='unknown'
- `tests/conftest.py:42` call='os.environ.get', key='CI', reachability='test-only'
- `src/vulcan/tests/conftest.py:46` call='os.environ.get', key='CI', reachability='test-only'
- `tests/test_000_imports_sanity.py:48` call='os.environ.get', key='CI', reachability='test-only'
- `src/ai_providers.py:1107` call='os.environ.get', key='COHERE_API_KEY', reachability='unknown'
- `src/dqs_service.py:127` call='os.environ.get', key='CORS_ORIGINS', reachability='unknown'
- `src/pii_service.py:133` call='os.environ.get', key='CORS_ORIGINS', reachability='unknown'
- `src/api_gateway.py:140` call='os.environ.get', key='CORS_ORIGINS', reachability='unknown'
- `src/vulcan/security_fixes.py:458` call='os.environ.get', key='DEBUG', reachability='unknown'
- `src/audit_log.py:119` call='os.environ.get', key='DLT_ANCHOR_CRITICAL', reachability='unknown'
- `src/audit_log.py:126` call='os.environ.get', key='DLT_BATCH_SIZE', reachability='unknown'
- `src/audit_log.py:123` call='os.environ.get', key='DLT_RETRY_COUNT', reachability='unknown'
- `src/dqs_service.py:420` call='os.environ.get', key='DQS_HOST', reachability='unknown'
- `src/dqs_service.py:421` call='os.environ.get', key='DQS_PORT', reachability='unknown'
- `src/vulcan/llm/hybrid_executor.py:1199` call='os.environ.get', key='ENABLE_DISTILLATION', reachability='unknown'
- `src/governance/app.py:681` call='os.environ.get', key='FLASK_DEBUG', reachability='unknown'
- `src/vulcan/security_fixes.py:453` call='os.environ.get', key='FLASK_ENV', reachability='unknown'
- `src/governance/app.py:691` call='os.environ.get', key='FLASK_HOST', reachability='unknown'
- `src/governance/app.py:698` call='os.environ.get', key='FLASK_PORT', reachability='unknown'
- `src/governance/app.py:50` call='os.environ.get', key='FORCE_HTTPS', reachability='unknown'
- `src/api_gateway.py:90` call='os.environ.get', key='GATEWAY_MODE', reachability='unknown'
- `src/vulcan/memory/hierarchical.py:1656` call='os.environ.get', key='GITHUB_ACTIONS', reachability='unknown'
- ... 381 more in JSON

## Capability Claims

Count: 18

- `src/vulcan/orchestrator/agent_lifecycle.py:83` reachability='unknown', text='Agent capability types with hierarchical relationships\n    \n    AGENT POOL CONFIGURATION FIX: Added specialized reasoning engine capabilities\n    to enable prop'
- `src/vulcan/orchestrator/agent_pool.py:1289` reachability='unknown', text="\n        Assign agent with timeout and proper locking to prevent race conditions\n        FIXED: Won't hang if no agents available\n        Note: Triggers cleanup"
- `src/vulcan/orchestrator/agent_pool.py:1350` reachability='unknown', text=') with no available agents for capability '
- `src/vulcan/orchestrator/agent_pool.py:1368` reachability='unknown', text=') with no available agents for capability '
- `src/utils/faiss_config.py:141` reachability='unknown', text='✓ FAISS library imported successfully (CPU capability detection unavailable: '
- `src/vulcan/orchestrator/agent_scoring.py:148` reachability='unknown', text='\n    Get available agents that can handle the specified capabilities.\n\n    Args:\n        manager: AgentPoolManager instance\n        capabilities: List of capabi'
- `src/vulcan/orchestrator/agent_scoring.py:176` reachability='unknown', text='\n    Get the current capability distribution in the agent pool.\n\n    AGENT POOL CONFIGURATION FIX: This method provides observability into\n    which capabilitie'
- `src/utils/faiss_config.py:1` reachability='unknown', text='\nFAISS Configuration and Initialization Module\n\nHandles FAISS import with proper CPU capability detection and informative\nmessaging about instruction set usage '
- `src/memory/cost_optimizer.py:223` reachability='unknown', text="\n        Analyze current cost structure with defensive programming.\n\n        FIXED: Now provides fallback logic when memory system methods don't exist.\n        "
- `src/full_platform.py:3107` reachability='unknown', text='Capability unavailable in production'
- `src/vulcan/orchestrator/agent_scoring.py:20` reachability='unknown', text='\n    Assign an available agent with required capability.\n\n    Must be called with manager.lock held.\n\n    Args:\n        manager: AgentPoolManager instance\n     '
- `src/vulcan/runtime/errors.py:43` reachability='production-reachable', text='capability_unavailable'
- `src/vulcan/routing/agent_collaboration.py:599` reachability='unknown', text='AgentCapability not available, using string capability'
- `src/vulcan/runtime/output.py:1` reachability='production-reachable', text='The one canonical, capability-minimized output-language contract.\n\nThe supported surface is ``und`` strict rendering of bounded arithmetic,\nunknown, and clarifi'
- `src/utils/faiss_config.py:38` reachability='unknown', text='\n    Initialize FAISS with proper CPU capability detection and informative logging.\n\n    This function:\n    1. Attempts to import the FAISS library\n    2. Detec'
- `src/vulcan/orchestrator/agent_scoring.py:64` reachability='unknown', text="[AgentPool] No available agent for capability '"
- `src/vulcan/orchestrator/agent_scoring.py:69` reachability='unknown', text="[AgentPool] All agents busy. No agent available for capability '"
- `src/vulcan/planning.py:799` reachability='unknown', text='Check if a capability is available in current mode.'

## Ownership Graph

Count: 296

- `src/vulcan/persistence/alignment.py:` owner_domain='alignment', reachability='unknown'
- `src/vulcan/runtime/alignment.py:` owner_domain='alignment', reachability='production-reachable'
- `src/vulcan/safety/governance_alignment.py:` owner_domain='alignment', reachability='unknown'
- `src/vulcan/tests/test_governance_alignment.py:` owner_domain='alignment', reachability='test-only'
- `tests/persistence/test_alignment_registry_v2.py:` owner_domain='alignment', reachability='test-only'
- `tests/security/test_persistent_audit_alignment.py:` owner_domain='alignment', reachability='test-only'
- `tests/test_phase9b_alignment_bridge.py:` owner_domain='alignment', reachability='test-only'
- `tests/test_phase9d_alignment_review.py:` owner_domain='alignment', reachability='test-only'
- `scripts/audit_module_usage.py:` owner_domain='audit', reachability='research-only'
- `src/audit_log.py:` owner_domain='audit', reachability='unknown'
- `src/protocols/audit.py:` owner_domain='audit', reachability='unknown'
- `src/security_audit_engine.py:` owner_domain='audit', reachability='unknown'
- `src/vulcan/persistence/audit/__init__.py:` owner_domain='audit', reachability='unknown'
- `src/vulcan/persistence/audit/events.py:` owner_domain='audit', reachability='unknown'
- `src/vulcan/persistence/audit/index.py:` owner_domain='audit', reachability='unknown'
- `src/vulcan/persistence/audit/reconcile.py:` owner_domain='audit', reachability='unknown'
- `src/vulcan/runtime/audit.py:` owner_domain='audit', reachability='production-reachable'
- `src/vulcan/safety/rollback_audit.py:` owner_domain='audit', reachability='unknown'
- `src/vulcan/tests/test_rollback_audit.py:` owner_domain='audit', reachability='test-only'
- `tests/integration/test_memory_audit_composition.py:` owner_domain='audit', reachability='test-only'
- `tests/persistence/test_audit_semantics.py:` owner_domain='audit', reachability='test-only'
- `tests/persistence/test_segmented_audit.py:` owner_domain='audit', reachability='test-only'
- `tests/security/test_persistent_audit_alignment.py:` owner_domain='audit', reachability='test-only'
- `tests/test_audit_log.py:` owner_domain='audit', reachability='test-only'
- `tests/test_audit_protocol.py:` owner_domain='audit', reachability='test-only'
- `tests/test_security_audit_engine.py:` owner_domain='audit', reachability='test-only'
- `src/vulcan/world_model/meta_reasoning/csiu_enforcement.py:` owner_domain='csiu', reachability='unknown'
- `tests/test_csiu_enforcement_integration.py:` owner_domain='csiu', reachability='test-only'
- `tests/test_phase9_csiu.py:` owner_domain='csiu', reachability='test-only'
- `tests/test_phase9b_csiu_persistence.py:` owner_domain='csiu', reachability='test-only'
- `tests/test_phase9d_csiu_snapshot_state.py:` owner_domain='csiu', reachability='test-only'
- `src/vulcan/reasoning/analogical/domain_parser.py:` owner_domain='domain', reachability='unknown'
- `src/vulcan/reasoning/selection/tools/world_model_domain.py:` owner_domain='domain', reachability='unknown'
- `src/vulcan/reasoning/tool_domain_bridge.py:` owner_domain='domain', reachability='unknown'
- `src/vulcan/runtime/domain_registry.py:` owner_domain='domain', reachability='production-reachable'
- `src/vulcan/safety/domain_validators.py:` owner_domain='domain', reachability='unknown'
- `src/vulcan/semantic_bridge/domain_registry.py:` owner_domain='domain', reachability='unknown'
- `src/vulcan/tests/test_domain_registry.py:` owner_domain='domain', reachability='test-only'
- `src/vulcan/tests/test_domain_validators.py:` owner_domain='domain', reachability='test-only'
- `src/vulcan/tests/test_tool_domain_bridge.py:` owner_domain='domain', reachability='test-only'
- `src/vulcan/world_model/introspection_domain.py:` owner_domain='domain', reachability='unknown'
- `tests/persistence/test_domain_registry_v2.py:` owner_domain='domain', reachability='test-only'
- `tests/security/test_persistent_domain_registry.py:` owner_domain='domain', reachability='test-only'
- `tests/test_domain_specific_formatting.py:` owner_domain='domain', reachability='test-only'
- `src/vulcan/local_language/__init__.py:` owner_domain='language', reachability='unknown'
- `src/vulcan/local_language/adapter.py:` owner_domain='language', reachability='unknown'
- `src/vulcan/local_language/governance.py:` owner_domain='language', reachability='unknown'
- `src/vulcan/local_language/release.py:` owner_domain='language', reachability='unknown'
- `src/vulcan/local_language/tokenizer.py:` owner_domain='language', reachability='unknown'
- `src/vulcan/tests/test_language_interface_integration.py:` owner_domain='language', reachability='test-only'
- ... 246 more in JSON

## Text Entrypoints

### compose
- `docker-compose.dev.yml:76` image: ${MINIO_IMAGE:-minio/minio:latest}
- `docker-compose.dev.yml:79` command: server /data --console-address ":9001"
- `docker-compose.dev.yml:111` image: ${MINIO_MC_IMAGE:-minio/mc:latest}
- `docker-compose.dev.yml:116` entrypoint: >
- `docker-compose.dev.yml:139` image: postgres:14-alpine
- `docker-compose.dev.yml:157` command: postgres -c config_file=/etc/postgresql/postgresql.conf
- `docker-compose.dev.yml:171` image: redis:7-alpine
- `docker-compose.dev.yml:174` command: ["redis-server", "/usr/local/etc/redis/redis.conf"]
- `docker-compose.dev.yml:199` image: oliver006/redis_exporter:v1.62.0
- `docker-compose.dev.yml:222` image: quay.io/coreos/etcd:v3.5.5
- `docker-compose.dev.yml:232` command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
- `docker-compose.dev.yml:238` image: ${MILVUS_MINIO_IMAGE:-minio/minio:latest}
- `docker-compose.dev.yml:249` command: minio server /minio_data --console-address ":9001"
- `docker-compose.dev.yml:260` image: milvusdb/milvus:v2.4.6
- `docker-compose.dev.yml:554` image: openpolicyagent/opa:0.65.0
- `docker-compose.dev.yml:557` command:
- `docker-compose.dev.yml:695` image: nginx:1.27-alpine
- `docker-compose.dev.yml:758` image: prom/prometheus:v2.48.0
- `docker-compose.dev.yml:761` command:
- `docker-compose.dev.yml:792` image: grafana/grafana:10.2.2
- `docker-compose.dev.yml:827` image: jaegertracing/all-in-one:1.52
- `docker-compose.dev.yml:860` image: docker.elastic.co/elasticsearch/elasticsearch:8.11.3
- `docker-compose.dev.yml:897` image: docker.elastic.co/kibana/kibana:8.11.3
- `docker-compose.dev.yml:925` image: prom/alertmanager:v0.26.0
- `docker-compose.dev.yml:928` command:
- `docker-compose.dev.yml:949` image: dpage/pgadmin4:8.0
- `docker-compose.dev.yml:973` image: rediscommander/redis-commander:latest
- `docker-compose.dev.yml:992` image: portainer/portainer-ce:2.19.4
- `docker-compose.dev.yml:995` command: -H unix:///var/run/docker.sock
- `docker-compose.dev.yml:1011` image: mailhog/mailhog:v1.0.1
- `docker-compose.dev.yml:1026` image: nginx:1.27-alpine
- `docker-compose.prod.yml:58` image: postgres:14-alpine
- `docker-compose.prod.yml:86` image: redis:7-alpine
- `docker-compose.prod.yml:89` command: ["redis-server", "--requirepass", "${REDIS_PASSWORD:?REDIS_PASSWORD is required}"]
- `docker-compose.prod.yml:109` image: minio/minio:RELEASE.2025-01-10T00-00-00Z
- `docker-compose.prod.yml:112` command: server /data --console-address ":9001"
- `docker-compose.prod.yml:140` image: quay.io/coreos/etcd:v3.5.5
- `docker-compose.prod.yml:150` command:
- `docker-compose.prod.yml:172` image: milvusdb/milvus:v2.3.4
- `docker-compose.prod.yml:180` command: ["milvus", "run", "standalone"]
- `docker-compose.prod.yml:215` image: ${REGISTRY:-ghcr.io}/${IMAGE_NAME:-musicmonk42/vulcanami_llm}:${VERSION:-latest}
- `docker-compose.prod.yml:429` image: ${REGISTRY:-ghcr.io}/${IMAGE_NAME:-musicmonk42/vulcanami_llm}-api:${VERSION:-latest}
- `docker-compose.prod.yml:487` image: ${REGISTRY:-ghcr.io}/${IMAGE_NAME:-musicmonk42/vulcanami_llm}-dqs:${VERSION:-latest}
- `docker-compose.prod.yml:529` image: ${REGISTRY:-ghcr.io}/${IMAGE_NAME:-musicmonk42/vulcanami_llm}-pii:${VERSION:-latest}
- `docker-compose.prod.yml:568` image: prom/prometheus:v2.48.0
- `docker-compose.prod.yml:571` command:
- `docker-compose.prod.yml:594` image: grafana/grafana:10.2.2
- `docker-compose.prod.yml:627` image: nginx:1.27-alpine

### docker
- `Dockerfile:312` CMD curl -fsS http://localhost:${PORT:-8000}/health/live || exit 1
- `Dockerfile:315` ENTRYPOINT ["/app/entrypoint.sh"]
- `Dockerfile:321` CMD ["sh", "-c", "uvicorn vulcan.runtime.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
- `docker/api/Dockerfile:135` CMD curl -fsS http://localhost:8000/health/live || exit 1
- `docker/api/Dockerfile:137` ENTRYPOINT ["/app/entrypoint.sh"]
- `docker/api/Dockerfile:140` CMD ["python", "-m", "uvicorn", "src.api_gateway:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
- `docker/dqs/Dockerfile:121` CMD curl -fsS http://localhost:8080/health || exit 1
- `docker/dqs/Dockerfile:123` ENTRYPOINT ["/app/entrypoint.sh"]
- `docker/dqs/Dockerfile:126` CMD ["python", "-m", "uvicorn", "src.dqs_service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
- `docker/pii/Dockerfile:120` CMD curl -fsS http://localhost:8082/health || exit 1
- `docker/pii/Dockerfile:122` ENTRYPOINT ["/app/entrypoint.sh"]
- `docker/pii/Dockerfile:125` CMD ["python", "-m", "uvicorn", "src.pii_service:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "1"]

### helm
- `helm/vulcanami/templates/deployment.yaml:34` image: {{ include "vulcanami.image" . | quote }}
- `helm/vulcanami/templates/deployment.yaml:36` command:
- `helm/vulcanami/templates/deployment.yaml:71` image: {{ include "vulcanami.image" . | quote }}
- `helm/vulcanami/templates/deployment.yaml:73` command:
- `helm/vulcanami/templates/deployment.yaml:76` args:
- `helm/vulcanami/templates/deployment.yaml:90` livenessProbe:
- `helm/vulcanami/templates/deployment.yaml:94` readinessProbe:
- `helm/vulcanami/values.yaml:13` image:
- `helm/vulcanami/values.yaml:23` # To get the digest of an image:
- `helm/vulcanami/values.yaml:117` livenessProbe:
- `helm/vulcanami/values.yaml:126` readinessProbe:
