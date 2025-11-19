📊 Graphix & VULCAN-AI Visualization Guide  
Version: 2.1.0  
Date: September 08, 2025

This guide details the setup and interpretation of visual outputs for the Graphix/VULCAN-AI platform. Visualizing the system's operational metrics is crucial for monitoring its health, performance, and ethical alignment during development. The primary tools are Prometheus for metrics collection and Grafana for dashboarding.

---

## 1. ⚙️ Setup Prometheus

Prometheus is an open-source monitoring system that scrapes and stores time-series data. The Graphix Arena exposes its metrics in a Prometheus-compatible format, making it easy to collect them.

Download and run Prometheus (e.g., v2.54.0) from the official website.

Configure prometheus.yml to scrape the Graphix Arena's metrics endpoint:

```yaml
scrape_configs:
  - job_name: 'graphix_arena'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

Start the Prometheus server:

```bash
./prometheus --config.file=prometheus.yml
```

Access: http://localhost:9090  
Expected: You should see the graphix_arena target in the "Targets" section with a "UP" state. You can query for metrics like execution_latency_seconds_bucket.

---

## 2. 📈 Setup Grafana

Grafana is a visualization tool that turns time-series data from Prometheus into dashboards. The ObservabilityManager can automatically generate a pre-configured dashboard for you.

Download and run Grafana (e.g., v11.2.0) from the official website.

Start the Grafana server:

```bash
./bin/grafana-server
```

Access: http://localhost:3000 (default login: admin/admin).

Configure Grafana:

- Add a Data Source: Connect Grafana to your running Prometheus server (URL: http://localhost:9090).
- Import the Dashboard: 
  - **Mission Control Dashboard (Recommended)**: Go to Dashboards -> Import and upload `dashboards/grafana/graphix_mission_control.json` for a comprehensive pre-built dashboard with panels for latency, errors, safety, energy, cache, and health metrics.
  - **Auto-generated Dashboard**: Alternatively, the ObservabilityManager creates a dashboard file at `observability_logs/graphix_dashboard.json`.

### Mission Control Dashboard

The pre-built Mission Control dashboard (`dashboards/grafana/graphix_mission_control.json`) includes:

- **95th Percentile Execution Latency**: Key SLO metric for Arena and VULCAN performance
- **Execution Error Rate**: HTTP 5xx errors and exposed error counters
- **Safety & Bias Detections**: Count of safety violations and bias detections
- **Energy/Cost Proxy**: Energy consumption tracking (if `graphix_energy_nj_total` is exposed)
- **Cache Hit Rate**: Graph execution and compilation cache efficiency
- **Agent Task Completion Rate**: Successful task completions over time
- **Service Health Status**: Real-time health indicators for all services

The dashboard supports:
- Instance templating (filter by Prometheus instance)
- 5-second auto-refresh
- Comprehensive alerting thresholds

---

## 3. 🗺️ Plot a Semantic Map

The ObservabilityManager can also generate graphviz plots to visualize the attention weights within a tensor, which is useful for interpreting model behavior.

Install Graphviz:

```bash
pip install graphviz
```

Generate a Plot:  
Run the following Python script to create a sample plot.

```python
from src.observability_manager import ObservabilityManager
import numpy as np

obs = ObservabilityManager()

# Create a sample 5x5 attention tensor
attention_tensor = np.random.rand(5, 5)

# Generate and save the plot
image_path = obs.plot_semantic_map(attention_tensor, labels=['Feat1', 'Feat2', 'Feat3', 'Feat4', 'Feat5'])

if image_path:
    print(f"Semantic map plot saved to: {image_path}")
```

Expected Output: A PNG file will be saved in the observability_logs/ directory showing a directed graph where edge thickness represents the attention weight between features.

---

## 4. 📊 Expected Outputs & Interpretation

Your monitoring setup will provide insights into the platform's operation.

### Grafana Dashboard

The auto-generated dashboard includes the following key panels:

| Panel Title                      | Metric Type    | What it Shows                                                                              |
|----------------------------------|---------------|--------------------------------------------------------------------------------------------|
| 95th Percentile Execution Latency| Histogram     | The latency (in seconds) that 95% of requests are faster than. Key SLO metric for performance.|
| Execution Error Rate             | Counter (Rate)| Number of errors per second, broken down by component. Dashboard includes an alert if rate exceeds 0.1/sec.|
| Total Audit Events               | Stat          | Count of total audit events logged, categorized by type (e.g., test_event, validation_pass).|
| Total Bias Detections            | Stat          | Counter tracking how many times the NSOAligner has detected a biased or risky proposal.     |
| Tensor Semantics                 | Table         | Shows the latest explainability scores for processed tensors.                               |

---

### Other Visuals

- **Graphviz Plots:** These plots show relationships within a tensor. Thicker lines between nodes indicate higher attention, helping developers understand which parts of an input are most influential on an output.
- **Slack Alerts:** If you configure the SLACK_BOT_TOKEN in your .env file, the SecurityAuditEngine will post real-time alerts to the specified channel for critical events like bias detections.