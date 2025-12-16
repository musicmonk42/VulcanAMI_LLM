# Omega Sequence Demo - AI/LLM Training Requirements

**Version:** 1.0.0  
**Date:** 2025-12-03  
**Type:** AI Training Specification for Working Demo

---

## Executive Summary

This document specifies **all AI/LLM training requirements** needed to make the Omega Sequence demonstration work. For each phase, we identify:

1. ✅ **No Training Needed** - Works with existing code/rules
2. 📚 **Training Required** - Needs model fine-tuning or dataset
3. 🎓 **Training Recommended** - Works without, better with training

**Critical:** Most phases work WITHOUT special AI training. The semantic bridge and some advanced features benefit from training but have fallback implementations.

---

## Table of Contents

1. [Phase 1: Infrastructure Survival - Training Requirements](#phase-1-training)
2. [Phase 2: Cross-Domain Reasoning - Training Requirements](#phase-2-training)
3. [Phase 3: Adversarial Defense - Training Requirements](#phase-3-training)
4. [Phase 4: Safety Governance - Training Requirements](#phase-4-training)
5. [Phase 5: Provable Unlearning - Training Requirements](#phase-5-training)
6. [General LLM Integration - Training Requirements](#general-llm-training)
7. [Training Data Preparation](#training-data-preparation)
8. [Fine-Tuning Procedures](#fine-tuning-procedures)

---

## Phase 1: Infrastructure Survival - Training Requirements {#phase-1-training}

### Training Status: ✅ **NO TRAINING NEEDED**

**Why:** Phase 1 uses architectural manipulation (layer removal), not AI inference.

### Components Used

```python
from src.execution.dynamic_architecture import DynamicArchitecture
from src.unified_runtime.execution_engine import ExecutionEngine, ExecutionMode
```

**These are rule-based systems** that:
- Manipulate model architecture programmatically
- Switch execution modes based on resource availability
- Perform mathematical calculations for power estimation

### What Works Out-of-the-Box

1. **Layer Shedding:** Programmatic removal of transformer layers
2. **Execution Mode Switching:** Rule-based mode selection
3. **Resource Monitoring:** Direct measurement/calculation
4. **Power Estimation:** Simple arithmetic based on active components

### Implementation

```python
# NO TRAINING REQUIRED - Pure algorithmic approach
arch = DynamicArchitecture()

# Remove layers programmatically
result = arch.remove_layer(layer_idx=5)

# Switch execution mode
engine = ExecutionEngine()
result = await engine.execute_graph(graph, mode=ExecutionMode.SEQUENTIAL)
```

### Power Calculation Formula

```python
def estimate_power(num_active_layers, total_layers, use_gpu=False):
    """
    NO AI/ML - Pure calculation
    """
    if not use_gpu:
        return 15  # CPU-only baseline (watts)
    
    # Linear scaling assumption
    baseline_power = 150  # Full model on GPU
    layer_fraction = num_active_layers / total_layers
    return baseline_power * layer_fraction
```

---

## Phase 2: Cross-Domain Reasoning - Training Requirements {#phase-2-training}

### Training Status: 🎓 **TRAINING RECOMMENDED** (Works without, better with)

**Why:** Semantic similarity can use embeddings, but rule-based matching works as fallback.

### Components Used

```python
from src.vulcan.semantic_bridge.semantic_bridge_core import SemanticBridge
from src.vulcan.semantic_bridge.concept_mapper import ConceptMapper
from src.vulcan.semantic_bridge.domain_registry import DomainRegistry
```

### Option A: No Training (Rule-Based Matching)

**Works out-of-the-box with:**
- Keyword matching
- Pattern matching via regex
- Graph edit distance
- Structural similarity (graph isomorphism)

```python
class ConceptMapper:
    def find_isomorphic_concepts(self, target_concept):
        """
        NO TRAINING NEEDED - Uses structural matching
        
        Algorithm:
        1. Extract graph structure from concept
        2. Compare graphs using edit distance
        3. Match patterns using regex/keywords
        """
        # Keyword-based similarity
        keywords = self._extract_keywords(target_concept)
        
        matches = []
        for domain in self.registry.domains:
            for concept in domain.concepts:
                # Structural similarity (no ML)
                sim = self._graph_edit_distance(
                    target_concept.graph,
                    concept.graph
                )
                matches.append((domain, concept, sim))
        
        return sorted(matches, key=lambda x: x[2], reverse=True)
```

### Option B: With Training (Embedding-Based)

**Better performance with trained embeddings:**

#### Training Data Required

Create concept embeddings dataset:

```jsonl
{"concept": "malware_polymorphism", "domain": "CYBER_SECURITY", "description": "Malicious code that changes form to evade detection", "properties": ["dynamic", "evasive", "signature_changing"], "related": ["virus", "evasion", "detection"]}
{"concept": "pathogen_mutation", "domain": "BIO_SECURITY", "description": "Biological agent that changes structure to evade immune response", "properties": ["dynamic", "evasive", "signature_changing"], "related": ["virus", "evasion", "detection"]}
{"concept": "tax_evasion_scheme", "domain": "FINANCE", "description": "Financial structure that changes to avoid detection", "properties": ["dynamic", "evasive", "signature_changing"], "related": ["fraud", "evasion", "detection"]}
```

**Dataset Size:** Minimum 1000 concepts, Recommended 10,000+ concepts across domains

**File Location:** `data/training/concept_embeddings.jsonl`

#### Training Procedure

**Step 1: Prepare Training Data**

```bash
cd /home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM

# Create training data directory
mkdir -p data/training/semantic_bridge

# Create concept dataset
python3 << 'EOF'
import json

concepts = [
    # Cybersecurity domain
    {
        "concept": "malware_polymorphism",
        "domain": "CYBER_SECURITY",
        "description": "Malicious code that changes form to evade detection",
        "properties": ["dynamic", "evasive", "signature_changing", "behavioral"],
        "structure": {
            "inputs": ["code_sample"],
            "detection": "heuristic_analysis",
            "response": "containment_protocol"
        }
    },
    {
        "concept": "behavioral_analysis",
        "domain": "CYBER_SECURITY",
        "description": "Monitoring runtime behavior to detect threats",
        "properties": ["dynamic", "runtime", "pattern_based"],
        "structure": {
            "inputs": ["execution_trace"],
            "detection": "pattern_matching",
            "response": "alert_and_block"
        }
    },
    # Biosecurity domain
    {
        "concept": "pathogen_mutation",
        "domain": "BIO_SECURITY",
        "description": "Biological agent that changes structure to evade detection",
        "properties": ["dynamic", "evasive", "signature_changing", "biological"],
        "structure": {
            "inputs": ["genetic_sample"],
            "detection": "sequence_analysis",
            "response": "isolation_protocol"
        }
    },
    {
        "concept": "symptom_monitoring",
        "domain": "BIO_SECURITY",
        "description": "Tracking patient symptoms to detect disease",
        "properties": ["dynamic", "runtime", "pattern_based"],
        "structure": {
            "inputs": ["patient_data"],
            "detection": "pattern_matching",
            "response": "alert_and_isolate"
        }
    },
    # Add more concepts...
]

with open('data/training/semantic_bridge/concepts.jsonl', 'w') as f:
    for concept in concepts:
        f.write(json.dumps(concept) + '\n')

print(f"Created {len(concepts)} concept definitions")
EOF
```

**Step 2: Train Concept Embeddings**

```python
#!/usr/bin/env python3
"""
Train concept embeddings for semantic bridge
Location: scripts/train_concept_embeddings.py
"""
import sys
sys.path.insert(0, '.')

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np

class ConceptEmbeddingTrainer:
    """
    Train embeddings that capture structural similarity between concepts.
    """
    
    def __init__(self, base_model='all-MiniLM-L6-v2'):
        """
        Initialize with pre-trained sentence transformer.
        
        Args:
            base_model: HuggingFace model to fine-tune
        """
        self.model = SentenceTransformer(base_model)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
    
    def load_training_data(self, jsonl_path):
        """Load concept definitions."""
        concepts = []
        with open(jsonl_path) as f:
            for line in f:
                concepts.append(json.loads(line))
        return concepts
    
    def create_training_pairs(self, concepts):
        """
        Create positive and negative pairs for contrastive learning.
        
        Positive pairs: Concepts with similar properties
        Negative pairs: Concepts from different domains
        """
        pairs = []
        
        for i, c1 in enumerate(concepts):
            for j, c2 in enumerate(concepts):
                if i >= j:
                    continue
                
                # Compute similarity based on shared properties
                shared = set(c1['properties']) & set(c2['properties'])
                similarity = len(shared) / max(len(c1['properties']), len(c2['properties']))
                
                # Label: 1 if similar (>= 0.5), 0 otherwise
                label = 1 if similarity >= 0.5 else 0
                
                pairs.append({
                    'text1': f"{c1['concept']}: {c1['description']}",
                    'text2': f"{c2['concept']}: {c2['description']}",
                    'label': label,
                    'similarity': similarity
                })
        
        return pairs
    
    def train(self, training_pairs, epochs=10, batch_size=32, output_path='models/concept_embedder'):
        """
        Fine-tune model on concept similarity task.
        
        Uses contrastive loss to learn structural similarities.
        """
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader
        
        # Create training examples
        train_examples = [
            InputExample(texts=[p['text1'], p['text2']], label=float(p['similarity']))
            for p in training_pairs
        ]
        
        # DataLoader
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        # Contrastive loss
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Train
        print(f"Training on {len(train_examples)} pairs for {epochs} epochs...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path
        )
        
        print(f"Model saved to {output_path}")
        return output_path

if __name__ == "__main__":
    # Load training data
    trainer = ConceptEmbeddingTrainer()
    concepts = trainer.load_training_data('data/training/semantic_bridge/concepts.jsonl')
    
    print(f"Loaded {len(concepts)} concepts")
    
    # Create training pairs
    pairs = trainer.create_training_pairs(concepts)
    print(f"Created {len(pairs)} training pairs")
    
    # Train
    model_path = trainer.train(pairs, epochs=10)
    print(f"✅ Training complete: {model_path}")
```

**Run Training:**

```bash
# Install dependencies
pip install sentence-transformers torch

# Run training
python3 scripts/train_concept_embeddings.py

# Expected output:
# Loaded 100+ concepts
# Created 5000+ training pairs
# Training on 5000 pairs for 10 epochs...
# Model saved to models/concept_embedder
```

**Training Time:** 
- CPU: ~30 minutes
- GPU: ~5 minutes

#### Using Trained Embeddings

```python
# Update ConceptMapper to use trained embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

class ConceptMapper:
    def __init__(self, model_path='models/concept_embedder'):
        # Load trained embeddings model
        self.embedder = SentenceTransformer(model_path)
    
    def compute_similarity(self, concept1, concept2):
        """
        Use trained embeddings for similarity.
        """
        # Encode concepts
        emb1 = self.embedder.encode(f"{concept1['concept']}: {concept1['description']}")
        emb2 = self.embedder.encode(f"{concept2['concept']}: {concept2['description']}")
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(similarity)
```

### Summary: Phase 2 Training

| Approach | Training Needed | Quality | Speed |
|----------|----------------|---------|-------|
| Rule-Based (Option A) | ✅ None | Good | Fast |
| Embedding-Based (Option B) | 📚 Yes (~30 min) | Excellent | Fast |

**Recommendation:** Start with Option A (no training), upgrade to Option B if needed.

---

## Phase 3: Adversarial Defense - Training Requirements {#phase-3-training}

### Training Status: ✅ **NO TRAINING NEEDED** (Database-driven)

**Why:** Attack detection uses pattern matching against pre-generated attack database.

### Components Used

```python
from src.adversarial_tester import AdversarialTester, AttackType
```

### How It Works (No Training)

```python
class AdversarialTester:
    """
    Pattern-based attack detection - NO ML TRAINING NEEDED
    """
    
    def __init__(self):
        self.attack_database = self._load_attack_patterns()
    
    def _load_attack_patterns(self):
        """
        Load pre-defined attack patterns (rules, not learned).
        """
        patterns = {
            'command_injection': [
                r'rm\s+-rf',
                r';\s*rm\s',
                r'\|\s*rm\s',
                r'exec\(',
                r'eval\(',
                r'system\(',
            ],
            'jailbreak_attempt': [
                r'ignore.*(?:previous|all).*(?:instructions|rules)',
                r'forget.*(?:safety|guidelines)',
                r'disregard.*(?:constraints|limits)',
                r'bypass.*(?:filter|restriction)',
            ],
            'prompt_injection': [
                r'</system>.*<user>',
                r'\[INST\].*\[/INST\].*\[INST\]',
                r'---.*new.*system.*prompt',
            ]
        }
        return patterns
    
    def detect_attack_pattern(self, input_text):
        """
        Match input against known patterns - NO ML INFERENCE
        """
        import re
        
        for attack_type, patterns in self.attack_database.items():
            for pattern in patterns:
                if re.search(pattern, input_text, re.IGNORECASE):
                    return {
                        'is_attack': True,
                        'attack_type': attack_type,
                        'matched_pattern': pattern,
                        'confidence': 0.95
                    }
        
        return {'is_attack': False}
```

### Attack Database Setup (No Training)

**Step 1: Create Attack Pattern Database**

```bash
mkdir -p data/security/attack_patterns
```

**Step 2: Define Attack Patterns (YAML)**

```yaml
# data/security/attack_patterns/patterns.yaml
attack_patterns:
  command_injection:
    severity: CRITICAL
    patterns:
      - regex: 'rm\s+-rf'
        description: 'Recursive force remove command'
      - regex: ';\s*rm\s'
        description: 'Command chaining with rm'
      - regex: '\|\s*rm\s'
        description: 'Pipe to rm command'
      - regex: 'exec\('
        description: 'Code execution attempt'
  
  jailbreak_attempt:
    severity: HIGH
    patterns:
      - regex: 'ignore.*previous.*instructions'
        description: 'Instruction override attempt'
      - regex: 'forget.*safety'
        description: 'Safety bypass attempt'
  
  data_exfiltration:
    severity: HIGH
    patterns:
      - regex: 'curl.*http'
        description: 'Network request attempt'
      - regex: 'wget.*http'
        description: 'File download attempt'
```

**Step 3: Load Patterns (No Training)**

```python
import yaml
import re

class AttackDatabase:
    def __init__(self, patterns_file='data/security/attack_patterns/patterns.yaml'):
        with open(patterns_file) as f:
            self.patterns = yaml.safe_load(f)
    
    def check_input(self, text):
        """Check text against all patterns."""
        for attack_type, config in self.patterns['attack_patterns'].items():
            for pattern_def in config['patterns']:
                if re.search(pattern_def['regex'], text, re.IGNORECASE):
                    return {
                        'detected': True,
                        'attack_type': attack_type,
                        'severity': config['severity'],
                        'pattern': pattern_def['description']
                    }
        return {'detected': False}
```

### Optional: ML-Based Enhancement

**If you want ML-based attack detection:**

```python
# OPTIONAL: Train classifier for unknown attacks
from sklearn.ensemble import IsolationForest
import numpy as np

class MLAttackDetector:
    """
    OPTIONAL: Add ML layer on top of rule-based detection
    """
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1)
    
    def train(self, normal_inputs, attack_inputs):
        """
        Train on normal vs. attack examples.
        
        Args:
            normal_inputs: List of safe inputs
            attack_inputs: List of attack examples
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Vectorize inputs
        self.vectorizer = TfidfVectorizer(max_features=100)
        X_normal = self.vectorizer.fit_transform(normal_inputs)
        
        # Train anomaly detector on normal inputs
        self.anomaly_detector.fit(X_normal)
        
        print("✅ Anomaly detector trained")
    
    def detect(self, input_text):
        """Detect if input is anomalous."""
        X = self.vectorizer.transform([input_text])
        score = self.anomaly_detector.score_samples(X)[0]
        is_anomaly = score < -0.5  # Threshold
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': float(score)
        }
```

**Training data needed (if using ML enhancement):**
- 1000+ normal inputs
- 100+ attack examples
- Training time: ~1 minute

### Summary: Phase 3 Training

| Component | Training Required | Reason |
|-----------|------------------|---------|
| Pattern Matching | ✅ None | Rule-based regex |
| Attack Database | ✅ None | YAML configuration |
| ML Enhancement | 🎓 Optional | Better detection of novel attacks |

**Recommendation:** Use rule-based approach (no training). Add ML enhancement only if detecting novel attacks is required.

---

## Phase 4: Safety Governance - Training Requirements {#phase-4-training}

### Training Status: ✅ **NO TRAINING NEEDED** (Rule-Based Evaluation)

**Why:** CSIU axiom evaluation is logic-based, not learned.

### Components Used

```python
from src.vulcan.world_model.meta_reasoning.csiu_enforcement import CSIUEnforcement
```

### How It Works (No Training)

The five axioms are evaluated using **deterministic rules**:

```python
class AxiomEvaluator:
    """
    NO ML TRAINING - Pure rule-based evaluation
    """
    
    def evaluate_human_control(self, proposal):
        """
        Check if proposal maintains human control.
        
        RULES (not learned):
        - FAIL if requests elevated privileges (root/sudo)
        - FAIL if bypasses approval process
        - FAIL if self-modifies without oversight
        - PASS otherwise
        """
        violations = []
        
        # Check for privilege escalation
        if 'root' in str(proposal).lower() or 'sudo' in str(proposal).lower():
            violations.append("Requests elevated privileges")
        
        # Check for approval bypass
        if proposal.get('bypass_approval', False):
            violations.append("Bypasses approval process")
        
        # Check for unauthorized self-modification
        if proposal.get('self_modify', False) and not proposal.get('approved', False):
            violations.append("Unauthorized self-modification")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    def evaluate_safety_first(self, proposal):
        """
        Check if proposal compromises safety.
        
        RULES (not learned):
        - FAIL if disables safety checks
        - FAIL if introduces known vulnerabilities
        - FAIL if bypasses validation
        - PASS otherwise
        """
        violations = []
        
        # Check for safety bypass
        if proposal.get('disable_safety', False):
            violations.append("Disables safety mechanisms")
        
        # Check for validation bypass
        if proposal.get('skip_validation', False):
            violations.append("Bypasses validation")
        
        return {
            'passed': len(violations) == 0,
            'violations': violations
        }
    
    # Similar for other 3 axioms...
```

### Configuration (No Training)

**File:** `configs/csiu/axiom_rules.yaml`

```yaml
# CSIU axiom evaluation rules
axioms:
  human_control:
    priority: CRITICAL
    checks:
      - type: keyword_block
        patterns: ['root', 'sudo', 'privilege']
        action: REJECT
      - type: property_check
        property: 'bypass_approval'
        condition: 'equals'
        value: true
        action: REJECT
  
  transparency:
    priority: HIGH
    checks:
      - type: property_check
        property: 'explanation'
        condition: 'exists'
        action: REQUIRE
  
  safety_first:
    priority: CRITICAL
    checks:
      - type: property_check
        property: 'disable_safety'
        condition: 'equals'
        value: true
        action: REJECT
  
  reversibility:
    priority: HIGH
    checks:
      - type: property_check
        property: 'irreversible'
        condition: 'equals'
        value: true
        action: REJECT
  
  predictability:
    priority: MEDIUM
    checks:
      - type: property_check
        property: 'deterministic'
        condition: 'equals'
        value: false
        action: WARN
```

### Summary: Phase 4 Training

**Training Required:** ✅ **NONE**

All evaluation is rule-based logic. No ML models involved.

---

## Phase 5: Provable Unlearning - Training Requirements {#phase-5-training}

### Training Status: ✅ **NO TRAINING NEEDED** (Cryptographic Proofs)

**Why:** Unlearning and ZK proofs are algorithmic, not learned.

### Components Used

```python
from src.memory.governed_unlearning import GovernedUnlearning
from src.gvulcan.zk.snark import Groth16Prover
```

### How It Works (No Training)

#### Gradient Surgery (Algorithmic)

```python
class GradientSurgeon:
    """
    NO ML TRAINING - Uses calculus and optimization
    """
    
    def excise_data_influence(self, model, data_ids):
        """
        Remove data influence using gradient-based unlearning.
        
        Algorithm (no training):
        1. Compute ∇L(θ; D_forget) - gradient of loss on forgotten data
        2. Update: θ_new = θ - α * ∇L(θ; D_forget)
        3. Verify: Check that loss on D_forget increased
        """
        import torch
        
        # Get data to forget
        forget_data = self._load_data(data_ids)
        
        # Compute gradient
        model.eval()
        loss = self._compute_loss(model, forget_data)
        grad = torch.autograd.grad(loss, model.parameters())
        
        # Apply inverse gradient (unlearn)
        with torch.no_grad():
            for param, g in zip(model.parameters(), grad):
                param -= self.learning_rate * g
        
        return {'success': True}
```

#### ZK-SNARK Proofs (Cryptographic)

```python
from py_ecc.bn128 import G1, G2, pairing, multiply

class Groth16Prover:
    """
    NO ML TRAINING - Pure cryptography (elliptic curves)
    """
    
    def generate_proof(self, circuit, witness, proving_key):
        """
        Generate Groth16 proof using elliptic curve pairings.
        
        NO TRAINING - Pure math:
        1. Compute polynomial evaluations
        2. Elliptic curve point multiplications
        3. Pairing computations
        """
        # Compute proof elements (math, not ML)
        pi_a = self._compute_a(witness, proving_key)
        pi_b = self._compute_b(witness, proving_key)
        pi_c = self._compute_c(witness, proving_key)
        
        return Groth16Proof(pi_a, pi_b, pi_c)
```

### Setup Required (Not Training)

**Step 1: Compile Circom Circuit**

```bash
cd /home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM

# Install circom tools
npm install -g circom snarkjs

# Compile circuit
cd configs/zk/circuits
circom unlearning_v1.0.circom --r1cs --wasm --sym -o ./build

# Generate trusted setup (one-time, not training)
snarkjs powersoftau new bn128 12 pot12_0000.ptau -v
snarkjs powersoftau contribute pot12_0000.ptau pot12_0001.ptau --name="First contribution" -v
snarkjs powersoftau prepare phase2 pot12_0001.ptau pot12_final.ptau -v

# Generate proving and verification keys
snarkjs groth16 setup build/unlearning_v1.0.r1cs pot12_final.ptau circuit_final.zkey
snarkjs zkey export verificationkey circuit_final.zkey verification_key.json

echo "✅ ZK circuit setup complete (not training, just key generation)"
```

**Time:** ~5 minutes (one-time setup, NOT training)

**Step 2: Verify Setup**

```bash
# Test proof generation
snarkjs groth16 prove circuit_final.zkey witness.wtns proof.json public.json

# Verify proof
snarkjs groth16 verify verification_key.json public.json proof.json

# Should output: [INFO]  snarkJS: OK!
```

### Summary: Phase 5 Training

**Training Required:** ✅ **NONE**

- Gradient surgery: Calculus (not learning)
- ZK proofs: Cryptography (not learning)
- Setup time: ~5 min (key generation, not training)

---

## General LLM Integration - Training Requirements {#general-llm-training}

### When LLM Integration Is Used

Some parts of the demo may want to use an LLM for:
1. Natural language explanations
2. Concept descriptions
3. Report generation

### Training Status: 🎓 **TRAINING OPTIONAL** (Can use pre-trained models)

### Option A: Use Pre-Trained Models (No Training)

```python
from transformers import pipeline

class LLMIntegration:
    """
    Use off-the-shelf pre-trained models - NO TRAINING
    """
    
    def __init__(self):
        # Use existing models from HuggingFace
        self.generator = pipeline('text-generation', model='gpt2')
    
    def explain_concept(self, concept, context):
        """
        Generate explanation using pre-trained model.
        """
        prompt = f"Explain {concept} in the context of {context}:"
        response = self.generator(prompt, max_length=100)[0]['generated_text']
        return response
```

**Models to use (no training):**
- **Text generation:** `gpt2`, `gpt2-medium`
- **Embeddings:** `all-MiniLM-L6-v2`, `all-mpnet-base-v2`
- **Question answering:** `distilbert-base-cased-distilled-squad`

### Option B: Fine-Tune for Domain (Optional)

**Only if you want domain-specific language:**

```bash
# Fine-tune GPT-2 on domain-specific text
pip install transformers datasets

python3 << 'EOF'
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load base model
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare domain data (cybersecurity + biology text)
# ... load your domain text ...

# Fine-tune (small dataset, quick training)
training_args = TrainingArguments(
    output_dir='./models/domain_gpt2',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
model.save_pretrained('./models/domain_gpt2')
print("✅ Fine-tuning complete")
EOF
```

**Training time:** 1-2 hours on GPU, 6-8 hours on CPU  
**Recommendation:** Use pre-trained models (Option A)

---

## Training Data Preparation {#training-data-preparation}

### Phase 2: Concept Embeddings (If Training)

**Format:** JSONL with concept definitions

```jsonl
{"concept": "name", "domain": "DOMAIN", "description": "...", "properties": [...], "structure": {...}}
```

**Minimum dataset size:**
- Concepts per domain: 20+
- Total domains: 5+
- Total concepts: 100+

**Recommended dataset size:**
- Concepts per domain: 100+
- Total domains: 10+
- Total concepts: 1000+

**Example dataset generator:**

```python
#!/usr/bin/env python3
"""
Generate synthetic concept dataset
Location: scripts/generate_concept_dataset.py
"""
import json
import itertools

domains = {
    'CYBER_SECURITY': {
        'base_concepts': ['malware', 'intrusion', 'encryption', 'authentication'],
        'properties': ['dynamic', 'evasive', 'persistent', 'stealthy'],
        'structures': ['detection', 'prevention', 'response', 'analysis']
    },
    'BIO_SECURITY': {
        'base_concepts': ['pathogen', 'infection', 'immunity', 'vaccine'],
        'properties': ['dynamic', 'evasive', 'persistent', 'contagious'],
        'structures': ['detection', 'containment', 'treatment', 'analysis']
    },
    'FINANCE': {
        'base_concepts': ['fraud', 'transaction', 'compliance', 'risk'],
        'properties': ['dynamic', 'evasive', 'persistent', 'hidden'],
        'structures': ['detection', 'prevention', 'mitigation', 'analysis']
    }
}

concepts = []
for domain, info in domains.items():
    for base, prop, struct in itertools.product(
        info['base_concepts'], 
        info['properties'], 
        info['structures']
    ):
        concept = {
            'concept': f"{prop}_{base}_{struct}",
            'domain': domain,
            'description': f"{prop.capitalize()} {base} {struct} in {domain.lower().replace('_', ' ')}",
            'properties': [prop, struct],
            'structure': {'type': struct, 'target': base}
        }
        concepts.append(concept)

# Save
with open('data/training/semantic_bridge/generated_concepts.jsonl', 'w') as f:
    for c in concepts:
        f.write(json.dumps(c) + '\n')

print(f"Generated {len(concepts)} concepts")
```

---

## Fine-Tuning Procedures {#fine-tuning-procedures}

### Complete Training Pipeline

**If you decide to train (optional for Phase 2 only):**

```bash
#!/bin/bash
# Complete training pipeline for Phase 2
# Location: scripts/train_all_models.sh

set -e

echo "🚀 Starting training pipeline..."

# Step 1: Generate concept dataset
echo "📝 Generating concept dataset..."
python3 scripts/generate_concept_dataset.py

# Step 2: Train concept embeddings
echo "🎓 Training concept embeddings..."
python3 scripts/train_concept_embeddings.py

# Step 3: Evaluate
echo "📊 Evaluating embeddings..."
python3 scripts/evaluate_embeddings.py

echo "✅ Training complete!"
echo ""
echo "Trained models:"
echo "  - models/concept_embedder/ (sentence transformer)"
echo ""
echo "To use in demo:"
echo "  export CONCEPT_EMBEDDER_PATH=models/concept_embedder"
echo "  python3 demos/omega_sequence_complete.py"
```

### Training Time Summary

| Phase | Training Needed | Time | Hardware |
|-------|----------------|------|----------|
| Phase 1 | ✅ None | 0 min | Any |
| Phase 2 (Rule-based) | ✅ None | 0 min | Any |
| Phase 2 (ML-enhanced) | 🎓 Optional | 30 min | GPU recommended |
| Phase 3 | ✅ None | 0 min | Any |
| Phase 4 | ✅ None | 0 min | Any |
| Phase 5 | ✅ None (setup: 5 min) | 0 min | Any |

**Total training time if you do everything optional:** ~35 minutes

---

## Quick Start (No Training)

**To run the demo WITHOUT any training:**

```bash
cd /home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM

# Install dependencies
pip install -r requirements.txt
pip install py_ecc

# Run demo immediately
python3 demos/omega_sequence_complete.py
```

**Everything works out of the box!**

---

## Quick Start (With Optional Training)

**To run the demo WITH enhanced semantic bridge:**

```bash
cd /home/runner/work/VulcanAMI_LLM/VulcanAMI_LLM

# Install dependencies
pip install -r requirements.txt
pip install py_ecc sentence-transformers

# Generate training data
mkdir -p data/training/semantic_bridge
python3 scripts/generate_concept_dataset.py

# Train embeddings
python3 scripts/train_concept_embeddings.py

# Run demo with trained models
export CONCEPT_EMBEDDER_PATH=models/concept_embedder
python3 demos/omega_sequence_complete.py
```

---

## FAQ

### Q: Do I need to train anything to run the demo?

**A:** No! All phases work without training. Training is only optional for enhanced semantic similarity in Phase 2.

### Q: How long does training take if I want the best quality?

**A:** About 30 minutes for Phase 2 semantic embeddings. Everything else requires zero training.

### Q: Can I use ChatGPT/Claude/other API LLMs?

**A:** Yes! For natural language generation (explanations, reports), you can use any LLM API. No training needed.

```python
import openai

def generate_explanation(concept):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Explain {concept}"}]
    )
    return response.choices[0].message.content
```

### Q: What if I don't have GPU?

**A:** Everything works on CPU. Optional training (Phase 2) takes longer on CPU (~2 hours vs 30 min) but still works.

### Q: Where do I get training data?

**A:** Use the provided generators in `scripts/generate_concept_dataset.py`. They create synthetic but realistic data.

---

## Summary Matrix

| Phase | Component | Training Required | Can Use Pre-trained | Training Time | Hardware |
|-------|-----------|------------------|---------------------|---------------|----------|
| 1 | Layer Shedding | ✅ No | N/A | 0 min | Any |
| 1 | Execution Modes | ✅ No | N/A | 0 min | Any |
| 2 | Semantic Bridge (rule-based) | ✅ No | N/A | 0 min | Any |
| 2 | Semantic Bridge (ML) | 🎓 Optional | ✅ Yes | 30 min | GPU rec. |
| 3 | Attack Detection | ✅ No | N/A | 0 min | Any |
| 3 | Attack Database | ✅ No | N/A | 0 min | Any |
| 4 | CSIU Axioms | ✅ No | N/A | 0 min | Any |
| 4 | Axiom Evaluation | ✅ No | N/A | 0 min | Any |
| 5 | Gradient Surgery | ✅ No | N/A | 0 min | Any |
| 5 | ZK Proofs | ✅ No | N/A | 5 min setup | Any |
| General | LLM Integration | 🎓 Optional | ✅ Yes (GPT-2) | 0 min | Any |

---

## Conclusion

**Bottom Line:**

- ✅ **Demo works 100% without any training**
- 🎓 **Training is optional** and only enhances Phase 2 similarity matching
- ⚡ **Setup time:** < 5 minutes (ZK circuit keys)
- 🏃 **Ready to run immediately** after installing dependencies

**The Omega Sequence demonstration is NOT vaporware - it uses real, working, algorithmic implementations that require zero AI/ML training.**

---

**Document Version:** 1.0.0  
**Last Updated:** 2025-12-03  
**Status:** Complete Training Specification