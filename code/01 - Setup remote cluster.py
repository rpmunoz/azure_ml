import os

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

configFolder = '../config'

print("Setting up service principal")
sp = ServicePrincipalAuthentication(tenant_id=os.environ['AML_TENANT_ID'],
                                    service_principal_id=os.environ['AML_PRINCIPAL_ID'],
                                    service_principal_password=os.environ['AML_PRINCIPAL_PASS'])

print('Setting up workspace')
ws = Workspace.from_config(path=configFolder, auth=sp)

# Create a compute target
clusterName = "cpucluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=clusterName)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4)

    compute_target = ComputeTarget.create(ws, clusterName, compute_config)

    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

    # For a more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())
