\# Upgraded Modules - Usage Guide



This document provides comprehensive documentation for the upgraded `governed\_unlearning.py` and `cost\_optimizer.py` modules.



---



\## 1. Governed Unlearning System



\### Overview

The `GovernedUnlearning` system provides enterprise-grade memory unlearning with governance controls, audit trails, and zero-knowledge proofs.



\### Key Features



\#### 🎯 Core Capabilities

\- \*\*Multi-stakeholder governance\*\* with voting mechanisms

\- \*\*Multiple unlearning methods\*\* (gradient surgery, exact removal, retraining, etc.)

\- \*\*Urgency-based prioritization\*\* (Critical, High, Normal, Low)

\- \*\*Zero-knowledge proof generation\*\* for verification

\- \*\*Comprehensive audit trails\*\* with persistent logging

\- \*\*Conflict resolution\*\* for overlapping patterns

\- \*\*Batch processing\*\* for multiple patterns

\- \*\*Rollback support\*\* via task cancellation

\- \*\*Real-time metrics tracking\*\*



\#### 🔐 Unlearning Methods

1\. \*\*Gradient Surgery\*\* - Default, balanced approach (≤6h)

2\. \*\*Exact Removal\*\* - Fast, precise deletion (≤2h)

3\. \*\*Retraining\*\* - Complete model retraining (≤24h)

4\. \*\*Cryptographic Erasure\*\* - Key-based deletion (≤1h)

5\. \*\*Differential Privacy\*\* - Privacy-preserving removal (≤8h)



\### Usage Examples



\#### Basic Usage

```python

from governed\_unlearning import GovernedUnlearning, UrgencyLevel, UnlearningMethod



\# Initialize

unlearning = GovernedUnlearning(

&nbsp;   persistent\_memory=memory\_system,

&nbsp;   consensus\_engine=consensus,  # Optional, uses SimpleConsensusEngine if not provided

&nbsp;   max\_workers=4,

&nbsp;   audit\_log\_file="/var/log/unlearning\_audit.jsonl"

)



\# Request unlearning

result = unlearning.request\_unlearning(

&nbsp;   pattern="user\_data\_12345",

&nbsp;   requester\_id="admin@company.com",

&nbsp;   urgency=UrgencyLevel.HIGH,

&nbsp;   method=UnlearningMethod.GRADIENT\_SURGERY,

&nbsp;   justification="GDPR right to be forgotten request",

&nbsp;   affected\_entities=\["user\_profile", "user\_messages"]

)



print(f"Status: {result.status.value}")

print(f"Proposal ID: {result.proposal\_id}")

```



\#### Batch Unlearning

```python

\# Unlearn multiple patterns at once

patterns = \[

&nbsp;   "user\_data\_12345",

&nbsp;   "user\_data\_67890",

&nbsp;   "user\_data\_11111"

]



results = unlearning.request\_batch\_unlearning(

&nbsp;   patterns=patterns,

&nbsp;   requester\_id="admin@company.com",

&nbsp;   urgency=UrgencyLevel.NORMAL,

&nbsp;   method=UnlearningMethod.EXACT\_REMOVAL

)



for result in results:

&nbsp;   print(f"Pattern: {result.proposal\_id} - Status: {result.status.value}")

```



\#### Task Monitoring

```python

\# Check task status

task = unlearning.get\_task\_status("task\_abc123")

if task:

&nbsp;   print(f"Progress: {task.progress \* 100:.1f}%")

&nbsp;   print(f"Status: {task.status.value}")

&nbsp;   if task.completed\_at:

&nbsp;       print(f"Duration: {task.get\_duration():.2f}s")



\# Cancel a task

if unlearning.cancel\_task("task\_abc123"):

&nbsp;   print("Task cancelled successfully")

```



\#### Verification

```python

\# Verify unlearning completed

verification = unlearning.verify\_unlearning("user\_data\_12345")

print(f"Is unlearned: {verification\['is\_unlearned']}")

print(f"Has residual: {verification\['has\_residual\_data']}")

print(f"Status: {verification\['status']}")

```



\#### Metrics and Audit

```python

\# Get metrics

metrics = unlearning.get\_metrics()

print(f"Total requests: {metrics.total\_requests}")

print(f"Success rate: {metrics.get\_success\_rate():.1%}")

print(f"Average execution time: {metrics.average\_execution\_time:.2f}s")

print(f"Total savings: {metrics.total\_patterns\_removed} patterns removed")



\# Get audit trail

audit\_entries = unlearning.get\_audit\_trail(n=50)

for entry in audit\_entries:

&nbsp;   print(f"{entry\['type']}: {entry\['timestamp']}")

```



\#### Custom Callbacks

```python

\# Add callbacks for events

def on\_approval(proposal, result):

&nbsp;   print(f"✅ Approved: {proposal.pattern}")

&nbsp;   # Send notification, update dashboard, etc.



def on\_completion(task):

&nbsp;   print(f"✅ Completed: {task.pattern}")

&nbsp;   # Log completion, trigger next steps, etc.



def on\_failure(task):

&nbsp;   print(f"❌ Failed: {task.pattern} - {task.error}")

&nbsp;   # Alert administrators, retry logic, etc.



unlearning.on\_approval.append(on\_approval)

unlearning.on\_completion.append(on\_completion)

unlearning.on\_failure.append(on\_failure)

```



\#### Custom Consensus Engine

```python

from governed\_unlearning import SimpleConsensusEngine



\# Configure consensus

consensus = SimpleConsensusEngine(

&nbsp;   auto\_approve\_threshold=0.67,  # 67% approval needed

&nbsp;   quorum\_required=3,             # Need 3 votes

&nbsp;   admin\_override\_enabled=True

)



\# Add voters

consensus.add\_voter("alice@company.com")

consensus.add\_voter("bob@company.com")

consensus.add\_voter("charlie@company.com")



\# Vote on proposal

result = consensus.vote("proposal\_123", "alice@company.com", approve=True)

result = consensus.vote("proposal\_123", "bob@company.com", approve=True)

\# Approval threshold met, proposal approved

```



\### Data Structures



\#### IRProposal

```python

@dataclass

class IRProposal:

&nbsp;   proposal\_id: str

&nbsp;   ir\_content: Dict\[str, Any]

&nbsp;   proposer\_id: str

&nbsp;   timestamp: float

&nbsp;   urgency: UrgencyLevel

&nbsp;   justification: str

&nbsp;   affected\_entities: List\[str]

&nbsp;   metadata: Dict\[str, Any]

```



\#### GovernanceResult

```python

@dataclass

class GovernanceResult:

&nbsp;   proposal\_id: str

&nbsp;   status: ProposalStatus

&nbsp;   details: Dict\[str, Any]

&nbsp;   votes: Dict\[str, bool]

&nbsp;   approval\_timestamp: Optional\[float]

&nbsp;   rejection\_reason: Optional\[str]

&nbsp;   execution\_result: Optional\[Dict\[str, Any]]

```



---



\## 2. Cost Optimizer



\### Overview

The `CostOptimizer` provides comprehensive cost analysis, optimization, and budget management for memory storage systems.



\### Key Features



\#### 💰 Core Capabilities

\- \*\*Multi-strategy optimization\*\* (Aggressive, Balanced, Conservative, etc.)

\- \*\*Cost analysis and breakdown\*\* by category

\- \*\*Budget management\*\* with alerts and hard limits

\- \*\*Automated scheduling\*\* for periodic optimization

\- \*\*What-if analysis\*\* for planning

\- \*\*Cost forecasting\*\* based on trends

\- \*\*Dry-run mode\*\* for safe testing

\- \*\*Comprehensive metrics tracking\*\*

\- \*\*Rollback support\*\* for failed optimizations



\#### 📊 Optimization Strategies

1\. \*\*Aggressive\*\* - Maximum savings, may impact performance

2\. \*\*Balanced\*\* - Good balance of cost and performance (default)

3\. \*\*Conservative\*\* - Minimal risk, only high-priority optimizations

4\. \*\*Cost-Focused\*\* - Maximize cost savings

5\. \*\*Performance-Focused\*\* - Maintain performance, optimize only where it helps



\### Usage Examples



\#### Basic Usage

```python

from cost\_optimizer import CostOptimizer, OptimizationStrategy, BudgetConfig



\# Configure budget

budget = BudgetConfig(

&nbsp;   monthly\_budget=1000.0,

&nbsp;   storage\_limit\_gb=1000.0,

&nbsp;   bandwidth\_limit\_gb=5000.0,

&nbsp;   alert\_threshold=0.8,    # Alert at 80%

&nbsp;   hard\_limit\_threshold=0.95  # Hard stop at 95%

)



\# Initialize optimizer

optimizer = CostOptimizer(

&nbsp;   persistent\_memory=memory\_system,

&nbsp;   budget\_config=budget,

&nbsp;   default\_strategy=OptimizationStrategy.BALANCED,

&nbsp;   auto\_optimize=True,      # Enable automatic optimization

&nbsp;   optimization\_interval=3600  # Run every hour

)



\# Run optimization

report = optimizer.optimize\_storage(

&nbsp;   strategy=OptimizationStrategy.BALANCED,

&nbsp;   dry\_run=False

)



print(f"Savings: ${report.savings:.2f}")

print(f"Savings %: {report.savings\_percentage:.1f}%")

print(f"Actions: {', '.join(report.actions\_taken)}")

```



\#### Dry Run (Safe Testing)

```python

\# Test optimization without making changes

report = optimizer.optimize\_storage(

&nbsp;   strategy=OptimizationStrategy.AGGRESSIVE,

&nbsp;   dry\_run=True

)



print("Recommendations:")

for rec in report.recommendations:

&nbsp;   print(f"  - {rec}")



print("\\nNo changes were made (dry run)")

```



\#### Full Optimization

```python

\# Optimize both storage and retrieval

reports = optimizer.optimize\_full(

&nbsp;   strategy=OptimizationStrategy.COST\_FOCUSED,

&nbsp;   dry\_run=False

)



for report in reports:

&nbsp;   print(f"\\n{report.optimization\_id}:")

&nbsp;   print(f"  Savings: ${report.savings:.2f}")

&nbsp;   print(f"  Duration: {report.get\_duration():.2f}s")

```



\#### Budget Management

```python

\# Check budget status

status = optimizer.check\_budget()

print(f"Current monthly cost: ${status\['current\_monthly\_cost']:.2f}")

print(f"Projected monthly: ${status\['projected\_monthly\_cost']:.2f}")

print(f"Budget limit: ${status\['budget\_limit']:.2f}")

print(f"Usage: {status\['usage\_percentage']:.1f}%")

print(f"Status: {status\['status']}")



\# Forecast future costs

forecast = optimizer.forecast\_costs(days=30)

print(f"\\nCost forecast (30 days):")

print(f"  Current daily: ${forecast\['current\_daily\_cost']:.2f}")

print(f"  Projected total: ${forecast\['projected\_total']:.2f}")

print(f"  Trend: {forecast\['trend']}")

```



\#### What-If Analysis

```python

\# Analyze potential changes

changes = {

&nbsp;   'reduce\_storage\_by\_gb': 200,      # Reduce storage by 200 GB

&nbsp;   'increase\_cdn\_hit\_rate': 0.15     # Improve CDN hit rate by 15%

}



analysis = optimizer.what\_if\_analysis(changes)

print(f"Current cost: ${analysis\['current\_cost']:.2f}")

print(f"Projected cost: ${analysis\['projected\_cost']:.2f}")

print(f"Potential savings: ${analysis\['potential\_savings']:.2f}")

print(f"Savings %: {analysis\['savings\_percentage']:.1f}%")

```



\#### Metrics and History

```python

\# Get optimization metrics

metrics = optimizer.get\_metrics()

print(f"Total optimizations: {metrics.total\_optimizations}")

print(f"Success rate: {metrics.get\_success\_rate():.1%}")

print(f"Total savings: ${metrics.total\_savings:.2f}")

print(f"GB saved: {metrics.total\_gb\_saved:.1f} GB")

print(f"Avg savings: {metrics.average\_savings\_percentage:.1f}%")



\# Get optimization history

history = optimizer.get\_optimization\_history(n=10)

for report in history:

&nbsp;   print(f"{report.optimization\_id}: ${report.savings:.2f} ({report.status.value})")

```



\#### Custom Callbacks

```python

\# Add callbacks for events

def on\_optimization\_complete(report):

&nbsp;   print(f"✅ Optimization complete: ${report.savings:.2f} saved")

&nbsp;   # Update dashboard, send notification, etc.



def on\_budget\_alert(level, status):

&nbsp;   print(f"⚠️  Budget alert ({level}): {status\['message']}")

&nbsp;   # Send alert to admins, trigger emergency optimizations, etc.



optimizer.on\_optimization\_complete.append(on\_optimization\_complete)

optimizer.on\_budget\_alert.append(on\_budget\_alert)

```



\#### Cost Analysis

```python

\# Analyze current costs

breakdown = optimizer.analyzer.analyze\_current\_costs()

print(f"Storage: ${breakdown.storage\_cost:.2f}")

print(f"Bandwidth: ${breakdown.bandwidth\_cost:.2f}")

print(f"Compute: ${breakdown.compute\_cost:.2f}")

print(f"API: ${breakdown.api\_cost:.2f}")

print(f"CDN: ${breakdown.cdn\_cost:.2f}")

print(f"Total: ${breakdown.total\_cost:.2f}")



\# Identify opportunities

opportunities = optimizer.analyzer.identify\_optimization\_opportunities()

for opp in opportunities:

&nbsp;   print(f"{opp\['type']} ({opp\['priority']}): {opp.get('potential\_savings\_gb', 0):.1f} GB")



\# Estimate savings

estimated = optimizer.analyzer.estimate\_savings(opportunities)

print(f"Total potential savings: ${estimated:.2f}")



\# Get cost trends

trends = optimizer.analyzer.get\_cost\_trends(days=30)

print(f"Trend: {trends\['trend']}")

print(f"Average cost: ${trends\['average\_cost']:.2f}")

print(f"Change: {trends\['change\_percentage']:.1f}%")

```



\### Data Structures



\#### CostBreakdown

```python

@dataclass

class CostBreakdown:

&nbsp;   storage\_cost: float

&nbsp;   bandwidth\_cost: float

&nbsp;   compute\_cost: float

&nbsp;   api\_cost: float

&nbsp;   cdn\_cost: float

&nbsp;   total\_cost: float

&nbsp;   timestamp: float

```



\#### OptimizationReport

```python

@dataclass

class OptimizationReport:

&nbsp;   optimization\_id: str

&nbsp;   strategy: OptimizationStrategy

&nbsp;   phase: OptimizationPhase

&nbsp;   started\_at: float

&nbsp;   completed\_at: Optional\[float]

&nbsp;   cost\_before: Optional\[CostBreakdown]

&nbsp;   cost\_after: Optional\[CostBreakdown]

&nbsp;   savings: float

&nbsp;   savings\_percentage: float

&nbsp;   actions\_taken: List\[str]

&nbsp;   recommendations: List\[str]

&nbsp;   warnings: List\[str]

&nbsp;   metadata: Dict\[str, Any]

```



\#### BudgetConfig

```python

@dataclass

class BudgetConfig:

&nbsp;   monthly\_budget: float = 1000.0

&nbsp;   storage\_limit\_gb: float = 1000.0

&nbsp;   bandwidth\_limit\_gb: float = 5000.0

&nbsp;   alert\_threshold: float = 0.8

&nbsp;   hard\_limit\_threshold: float = 0.95

```



---



\## Integration Examples



\### Combined Usage

```python

\# Initialize both systems

unlearning = GovernedUnlearning(

&nbsp;   persistent\_memory=memory\_system,

&nbsp;   audit\_log\_file="/var/log/unlearning.jsonl"

)



optimizer = CostOptimizer(

&nbsp;   persistent\_memory=memory\_system,

&nbsp;   budget\_config=BudgetConfig(monthly\_budget=5000.0),

&nbsp;   auto\_optimize=True

)



\# Callback: Optimize after unlearning completes

def optimize\_after\_unlearning(task):

&nbsp;   if task.status == ProposalStatus.COMPLETED:

&nbsp;       print("Unlearning completed, running optimization...")

&nbsp;       optimizer.optimize\_storage(strategy=OptimizationStrategy.BALANCED)



unlearning.on\_completion.append(optimize\_after\_unlearning)



\# Callback: Alert on budget issues

def handle\_budget\_alert(level, status):

&nbsp;   if level == 'critical':

&nbsp;       # Emergency optimization

&nbsp;       print("Critical budget alert! Running aggressive optimization...")

&nbsp;       optimizer.optimize\_full(strategy=OptimizationStrategy.AGGRESSIVE)



optimizer.on\_budget\_alert.append(handle\_budget\_alert)



\# Regular operations

result = unlearning.request\_unlearning(

&nbsp;   pattern="old\_data\_\*",

&nbsp;   requester\_id="system",

&nbsp;   urgency=UrgencyLevel.NORMAL

)



\# Check costs

budget\_status = optimizer.check\_budget()

if budget\_status\['status'] == 'warning':

&nbsp;   print("Budget warning - consider optimization")

```



\### Shutdown

```python

\# Clean shutdown

unlearning.shutdown()

optimizer.shutdown()

```



---



\## Advanced Features



\### Governed Unlearning



\#### Custom Urgency Thresholds

```python

unlearning.urgency\_thresholds = {

&nbsp;   UrgencyLevel.CRITICAL: timedelta(minutes=30),

&nbsp;   UrgencyLevel.HIGH: timedelta(hours=2),

&nbsp;   UrgencyLevel.NORMAL: timedelta(hours=12),

&nbsp;   UrgencyLevel.LOW: timedelta(days=3)

}

```



\#### Pattern Conflict Resolution

```python

\# Check for conflicts before requesting

conflicts = unlearning.\_check\_pattern\_conflicts("user\_data\_\*")

if conflicts:

&nbsp;   print(f"Warning: Conflicts with {len(conflicts)} existing patterns")

```



\### Cost Optimizer



\#### Custom Pricing

```python

\# Update pricing model

optimizer.analyzer.pricing = {

&nbsp;   'storage\_hot\_gb\_month': 0.30,

&nbsp;   'storage\_warm\_gb\_month': 0.12,

&nbsp;   'storage\_cold\_gb\_month': 0.03,

&nbsp;   'bandwidth\_gb': 0.10,

&nbsp;   'compute\_hour': 0.08,

&nbsp;   'api\_call': 0.0002,

&nbsp;   'cdn\_request': 0.000002

}

```



\#### Manual Optimization Planning

```python

\# Get opportunities

opportunities = optimizer.analyzer.identify\_optimization\_opportunities()



\# Create custom plan

plan = optimizer.\_create\_optimization\_plan(

&nbsp;   opportunities,

&nbsp;   OptimizationStrategy.COST\_FOCUSED

)



\# Review before execution

for step in plan:

&nbsp;   print(f"Will execute: {step\['description']}")

```



---



\## Best Practices



\### Governed Unlearning

1\. \*\*Always provide justification\*\* for unlearning requests

2\. \*\*Use appropriate urgency levels\*\* (don't overuse CRITICAL)

3\. \*\*Monitor audit trails\*\* regularly for compliance

4\. \*\*Verify unlearning\*\* after critical operations

5\. \*\*Set up callbacks\*\* for important events

6\. \*\*Use batch operations\*\* for multiple related patterns

7\. \*\*Configure consensus\*\* with appropriate stakeholders



\### Cost Optimizer

1\. \*\*Run dry-runs first\*\* before production optimizations

2\. \*\*Set realistic budgets\*\* with appropriate thresholds

3\. \*\*Review what-if analyses\*\* before major changes

4\. \*\*Monitor trends\*\* to catch cost increases early

5\. \*\*Use balanced strategy\*\* by default

6\. \*\*Schedule optimizations\*\* during low-traffic periods

7\. \*\*Test callbacks\*\* before deploying to production



---



\## Error Handling



Both modules include comprehensive error handling and logging. Monitor logs for issues:



```python

import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('governed\_unlearning')

logger = logging.getLogger('cost\_optimizer')

```



---



\## Performance Considerations



\- \*\*Governed Unlearning\*\*: Parallel execution with configurable max\_workers

\- \*\*Cost Optimizer\*\*: Automatic scheduling with configurable intervals

\- \*\*Both\*\*: Thread-safe operations with proper locking

\- \*\*Memory\*\*: Bounded history (deques with maxlen)

\- \*\*Graceful Shutdown\*\*: Both support clean shutdown procedures



---



\## License \& Support



These upgraded modules are production-ready implementations with enterprise-grade features. For support:

\- Review logs for detailed error messages

\- Check metrics for performance insights

\- Use dry-run mode for testing

\- Enable audit logging for compliance

