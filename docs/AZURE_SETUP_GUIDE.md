# Azure AKS Deployment Setup Guide

This guide provides a comprehensive walkthrough for setting up automated Azure Kubernetes Service (AKS) deployment via GitHub Actions.

## Quick Start

If you're experiencing "Az CLI Login failed" errors, follow these steps:

### 1. Create Azure Service Principal

```bash
# Login to Azure
az login

# Get your subscription ID
SUBSCRIPTION_ID=$(az account show --query id --output tsv)
echo "Subscription ID: $SUBSCRIPTION_ID"

# Create Service Principal with contributor role
az ad sp create-for-rbac \
  --name "github-actions-vulcanami" \
  --role contributor \
  --scopes /subscriptions/$SUBSCRIPTION_ID \
  --sdk-auth
```

**Expected Output:**
```json
{
  "clientId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "clientSecret": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "subscriptionId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "tenantId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  ...
}
```

### 2. Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to: **Settings** → **Secrets and variables** → **Actions** → **Repository secrets**
3. Click "New repository secret" and add each of the following:

| Secret Name | Value | Source |
|------------|-------|--------|
| `AZURE_CLIENT_ID` | Copy from `clientId` | Service Principal output |
| `AZURE_TENANT_ID` | Copy from `tenantId` | Service Principal output |
| `AZURE_SUBSCRIPTION_ID` | Copy from `subscriptionId` | Service Principal output |

### 3. Update Workflow Configuration

Edit `.github/workflows/azure-kubernetes-service-helm.yml` and update these environment variables:

```yaml
env:
  AZURE_CONTAINER_REGISTRY: "your-acr-name"           # Without .azurecr.io
  CONTAINER_NAME: "vulcanami-llm"                     # Your image name
  RESOURCE_GROUP: "vulcanami-prod"                    # Your resource group
  CLUSTER_NAME: "vulcanami-cluster"                   # Your AKS cluster
  CHART_PATH: "helm/vulcanami"                        # Path to Helm chart
  CHART_OVERRIDE_PATH: "helm/vulcanami/values-prod.yaml"  # Values file
```

### 4. Create Azure Resources

If you haven't already, create the required Azure resources:

```bash
# Set variables
RESOURCE_GROUP="vulcanami-prod"
ACR_NAME="vulcanamiregistry"
AKS_NAME="vulcanami-cluster"
LOCATION="eastus"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --sku Standard

# Create AKS cluster
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $AKS_NAME \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-managed-identity \
  --attach-acr $ACR_NAME \
  --generate-ssh-keys

# Get credentials for local access
az aks get-credentials --resource-group $RESOURCE_GROUP --name $AKS_NAME
```

## How It Works

### Workflow Triggers

The Azure deployment workflow runs when:
- Code is pushed to the `main` branch
- Manually triggered via GitHub Actions UI

### Workflow Jobs

**Job 1: buildImage**
1. Checks out code
2. Frees up disk space
3. **Validates Azure credentials** (NEW - fails fast if secrets missing)
4. Logs into Azure
5. Builds and pushes Docker image to ACR

**Job 2: deploy**
1. Checks out code
2. **Validates Azure credentials** (NEW - fails fast if secrets missing)
3. Logs into Azure
4. Configures kubelogin
5. Gets AKS cluster context
6. Bakes Kubernetes manifests with Helm
7. Deploys to AKS

### Credential Validation

The workflow now includes a validation step that checks if all required secrets are present:

```yaml
- name: Check Azure credentials
  run: |
    if [ -z "${{ secrets.AZURE_CLIENT_ID }}" ] || [ -z "${{ secrets.AZURE_TENANT_ID }}" ] || [ -z "${{ secrets.AZURE_SUBSCRIPTION_ID }}" ]; then
      echo "ERROR: Required Azure secrets are not configured."
      echo "Please set the following secrets in your repository:"
      echo "  - AZURE_CLIENT_ID"
      echo "  - AZURE_TENANT_ID"
      echo "  - AZURE_SUBSCRIPTION_ID"
      exit 1
    fi
    echo "✓ All required Azure secrets are configured"
```

This step runs **before** the Azure login attempt and provides a clear error message if any secrets are missing.

## Troubleshooting

### Error: "Required Azure secrets are not configured"

**Cause:** One or more of the required GitHub secrets is missing or empty.

**Solution:**
1. Verify all three secrets are set in GitHub repository settings
2. Check that the secret values don't have leading/trailing spaces
3. Ensure you copied the entire value from the Service Principal output

### Error: "Az CLI Login failed"

**Cause:** The Azure credentials are invalid or expired.

**Solution:**
1. Verify the Service Principal still exists: `az ad sp list --display-name github-actions-vulcanami`
2. Check the Service Principal has Contributor role on the subscription
3. Recreate the Service Principal if needed (see Step 1 above)
4. Update the GitHub secrets with new values

### Error: "Resource group not found"

**Cause:** The `RESOURCE_GROUP` environment variable doesn't match your actual resource group name.

**Solution:**
1. Check your resource groups: `az group list --output table`
2. Update the `RESOURCE_GROUP` value in the workflow file

### Error: "Container registry not found"

**Cause:** The `AZURE_CONTAINER_REGISTRY` environment variable is incorrect.

**Solution:**
1. List your ACRs: `az acr list --output table`
2. Update the `AZURE_CONTAINER_REGISTRY` value (without .azurecr.io)

### Self-Hosted Runners

If you're using self-hosted runners, uncomment the "Install Azure CLI" step in the workflow:

```yaml
- name: Install Azure CLI
  run: |
    curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

This step appears twice (once in each job). GitHub-hosted runners have Azure CLI pre-installed.

## Monitoring and Verification

### Check Workflow Status

1. Go to GitHub repository → **Actions** tab
2. Select "Build and deploy an app to AKS with Helm"
3. View the latest run
4. Check for the green checkmark (✓) on the "Check Azure credentials" step

### Verify Deployment

```bash
# Get AKS credentials
az aks get-credentials --resource-group vulcanami-prod --name vulcanami-cluster

# Check pods
kubectl get pods -n vulcanami

# Check services
kubectl get svc -n vulcanami

# View logs
kubectl logs -f deployment/vulcanami -n vulcanami
```

## Security Best Practices

1. **Rotate credentials regularly**: Regenerate Service Principal credentials every 90 days
2. **Use least privilege**: Ensure Service Principal only has necessary permissions
3. **Monitor access**: Enable Azure AD audit logs to track Service Principal usage
4. **Secure secrets**: Never commit secrets to code or logs
5. **Review permissions**: Periodically review Service Principal role assignments

## Additional Resources

- [Azure Service Principal Documentation](https://docs.microsoft.com/en-us/azure/active-directory/develop/app-objects-and-service-principals)
- [Azure AKS Documentation](https://docs.microsoft.com/en-us/azure/aks/)
- [GitHub Actions for Azure](https://github.com/Azure/Actions)
- [Connecting GitHub to Azure](https://docs.microsoft.com/en-us/azure/developer/github/connect-from-azure)

## Related Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- [CI_CD.md](CI_CD.md) - CI/CD pipeline documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [NEW_ENGINEER_SETUP.md](NEW_ENGINEER_SETUP.md) - New engineer onboarding

## Support

If you continue to experience issues after following this guide:

1. Check the workflow logs for specific error messages
2. Verify all prerequisites are met
3. Review the troubleshooting section above
4. Consult the related documentation
5. Open an issue with detailed error logs

---

**Last Updated:** 2025-12-16
**Applies To:** `.github/workflows/azure-kubernetes-service-helm.yml`
