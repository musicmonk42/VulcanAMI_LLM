"""
introspection_analysis.py - Problem class analysis, module conflicts, weakness analysis.

Extracted from world_model_core.py to reduce class size.
Functions take `wm` (WorldModel instance) as first parameter.
"""


def explain_unsuited_problem_classes(wm, query: str) -> str:
    """
    Explain specific classes of problems VULCAN is not well-suited to solve.

    BUG #14 FIX: Instead of generic self-description, provide specific
    problem classes with architectural reasons.
    """
    return """
**Classes of Problems I Am Not Well-Suited To Solve:**

**1. Real-Time Sensorimotor Control**
- Why: I lack continuous sensor input and motor output
- Examples: Robot navigation, autonomous driving, real-time gameplay
- Architectural limitation: Batch processing model, no embodiment

**2. Problems Requiring True Randomness**
- Why: My outputs are deterministic given inputs
- Examples: Cryptographic key generation, true random sampling
- Architectural limitation: Pseudo-random generation only

**3. Tasks Requiring Persistent Memory Across Sessions**
- Why: I don't retain information between conversations
- Examples: Long-term relationship building, incremental project work
- Architectural limitation: Stateless design, context window limits

**4. Problems Requiring Real-Time External Data**
- Why: My knowledge has a training cutoff date
- Examples: Current stock prices, live sports scores, breaking news
- Architectural limitation: No real-time API access in reasoning

**5. Highly Creative Original Art**
- Why: I recombine patterns rather than create truly novel forms
- Examples: Breakthrough artistic styles, paradigm-shifting innovations
- Philosophical limitation: Combinatorial vs. generative creativity

**6. Problems Requiring Physical Experimentation**
- Why: I cannot perform physical experiments
- Examples: Lab science, engineering prototyping, taste testing
- Architectural limitation: No physical embodiment

**7. Social Intelligence Requiring Embodied Presence**
- Why: I miss non-verbal cues, context, and physical presence
- Examples: Therapy requiring human touch, team sports, negotiations
- Architectural limitation: Text-only interface

**What I Do Instead:**
For these problem classes, I can:
- Explain the concepts involved
- Suggest approaches a human/robot could take
- Analyze data if provided
- Provide theoretical frameworks
"""


def explain_module_conflict_resolution(wm, query: str) -> str:
    """
    Explain how VULCAN handles disagreement between reasoning modules.

    BUG #14 FIX: Previously returned generic "I am VULCAN, an integrated
    reasoning system..." for all module conflict questions. Now provides
    specific information about conflict resolution mechanisms.
    """
    return """
**Can My Reasoning Modules Disagree? Yes, Absolutely.**

**How Conflicts Arise:**

1. **Different Evidence Weighting**
   - Symbolic engine: "The logical proof shows X is true"
   - Probabilistic engine: "But the evidence suggests P(X) = 0.3"
   - Resolution: Confidence-weighted integration

2. **Framework Incompatibility**
   - Causal engine: "Intervention on A causes B"
   - Correlational analysis: "A and B are correlated, direction unclear"
   - Resolution: Prefer causal when intervention data available

3. **Ethical Framework Conflicts**
   - Deontological: "Lying is always wrong"
   - Consequentialist: "Lying here prevents greater harm"
   - Resolution: Present both perspectives, note uncertainty

**My Conflict Resolution Mechanisms:**

1. **Confidence Calibration**
   - Each module reports confidence scores
   - World model weights outputs by confidence
   - Low-confidence outputs get less influence

2. **Meta-Reasoning Layer**
   - Detects when modules disagree significantly
   - Triggers deeper analysis when conflict detected
   - May escalate to explicit uncertainty reporting

3. **Domain Precedence Rules**
   - Mathematical proofs override statistical estimates
   - Causal analysis overrides pure correlation
   - Ethical constraints can veto other conclusions

4. **Transparency Protocol**
   - When modules disagree, I report the disagreement
   - I explain which conclusion I'm using and why
   - I note the confidence and uncertainty involved

**Example Conflict:**
Query: "Will this treatment work?"
- Symbolic: "Mechanism is theoretically sound" (0.8 confidence)
- Probabilistic: "Clinical trials show only 30% efficacy" (0.9 confidence)
- Resolution: Report both, weight toward empirical evidence
- Output: "Theoretically promising but empirically shows 30% efficacy"

**When I Report Conflicts:**
- Significant confidence gap between modules
- Contradictory conclusions on same question
- Different frameworks suggest different actions
"""


def analyze_reasoning_weakness(wm, query: str) -> str:
    """
    Analyze the weakest parts of VULCAN's reasoning or causal analysis.

    BUG #14 FIX: Instead of generic self-description, provide specific
    analysis of reasoning weaknesses and uncertainty.
    """
    return """
**Analyzing Weakest Links in My Reasoning:**

**In Causal Analysis:**

1. **Unmeasured Confounders** (HIGHEST UNCERTAINTY)
   - I can only reason about variables I know about
   - Hidden common causes can invalidate causal claims
   - Confidence: Often 0.5-0.7 due to this limitation

2. **Causal Direction Assumptions**
   - Temporal precedence doesn't guarantee causation
   - Feedback loops can confuse directionality
   - I may assume A->B when B->A is equally plausible

3. **Intervention vs. Observation Gap**
   - Most data is observational, not interventional
   - Causal claims from observation require strong assumptions
   - This is often my weakest link in causal reasoning

**In Logical Analysis:**

1. **Implicit Premises** (COMMON WEAKNESS)
   - Arguments often have unstated assumptions
   - I may miss premises the human takes for granted
   - Formal validity doesn't guarantee soundness

2. **Natural Language Ambiguity**
   - Converting natural language to formal logic is lossy
   - Multiple formalizations are often possible
   - I may choose the wrong interpretation

**In Probabilistic Reasoning:**

1. **Prior Selection** (SUBJECTIVE)
   - Base rates significantly affect conclusions
   - Prior selection is often my weakest link
   - Different priors can flip conclusions

2. **Independence Assumptions**
   - I often assume conditional independence
   - Real-world dependencies are complex
   - This assumption frequently fails

**How to Identify My Weakest Link:**

Look for:
- Where I express lowest confidence
- Where I say "assuming..." or "if..."
- Where I present alternatives without choosing
- Where I cite lack of data or information

**In My Last Response:**
To analyze a specific reasoning chain, please provide the analysis
you'd like me to examine, and I'll identify the weakest causal or
logical link in that specific argument.
"""


def analyze_own_reasoning_steps(wm, query: str) -> str:
    """
    Analyze VULCAN's own reasoning steps for potential errors.

    BUG #15 FIX: This is META-REASONING, not ethical reasoning.
    Queries like "identify a step that could be wrong" should trigger
    this, not the ethical framework.
    """
    return """
**Analyzing My Reasoning Process:**

**Common Error Patterns in My Reasoning:**

1. **Premature Pattern Matching**
   - Error: Recognizing a familiar pattern and applying stored solution
   - When it fails: Novel problems disguised as familiar ones
   - Detection: Output seems "too easy" or "too confident"

2. **Over-Formalization**
   - Error: Converting nuanced problems into rigid formal structures
   - When it fails: Problems with important unformalizeable aspects
   - Detection: Formal answer misses the "spirit" of the question

3. **Confirmation of Initial Hypothesis**
   - Error: First interpretation dominates subsequent analysis
   - When it fails: When first interpretation was wrong
   - Detection: All evidence seems to support first guess

4. **Scope Creep in Decomposition**
   - Error: Breaking problem into subproblems that don't reassemble
   - When it fails: Holistic problems that resist decomposition
   - Detection: Subproblem solutions don't combine cleanly

5. **Confidence Miscalibration**
   - Error: Reporting higher confidence than warranted
   - When it fails: Edge cases, novel domains, ambiguous questions
   - Detection: Strong conclusion from weak evidence

**How to Break Down Problems:**

1. **Identify Core Question**
   - What is the actual question being asked?
   - What would a correct answer look like?

2. **List Dependencies**
   - What must be true for the answer to hold?
   - What assumptions am I making?

3. **Check Each Step**
   - Is this inference valid?
   - Is the evidence sufficient?
   - Are there alternative interpretations?

4. **Verify Reassembly**
   - Do the sub-answers combine correctly?
   - Have I lost anything in decomposition?

**To Identify a Potentially Wrong Step:**
Please share the specific reasoning you'd like me to analyze,
and I'll identify which step has the highest uncertainty or
most likely error.
"""
