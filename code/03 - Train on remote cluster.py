import os
import shutil

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core import Workspace, Dataset, Run
from azureml.core import Experiment

#from azureml.core.dataset import Dataset
from azureml.core.compute import ComputeTarget
#from azureml.train.dnn import TensorFlow

from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.estimator import Estimator

configFolder = '../config'
experimentName = 'sklearn-mnist'
clusterName = 'cpucluster'
datasetName = 'mnist_dataset'
scriptFolder = 'train_scripts'
shutil.copy('utils.py', scriptFolder)

print("Setting up service principal")
sp = ServicePrincipalAuthentication(tenant_id=os.environ['AML_TENANT_ID'],
                                    service_principal_id=os.environ['AML_PRINCIPAL_ID'],
                                    service_principal_password=os.environ['AML_PRINCIPAL_PASS'])

print('Setting up workspace')
ws = Workspace.from_config(path=configFolder, auth=sp)

print('Setting up experiment')
exp = Experiment(workspace=ws, name=experimentName)

print('Setting up cluster')
compute_target = ComputeTarget(workspace=ws, name=clusterName)

print('Setting up dataset')
mnistFileDataset = Dataset.get_by_name(workspace=ws, name=datasetName)

print("Defining environment")
# to install required packages
env = Environment('sklearn')
#cd = CondaDependencies.create(pip_packages=['azureml-dataprep[pandas,fuse]>=1.1.14', 'azureml-defaults'], conda_packages = ['scikit-learn==0.22.1'])
cd = CondaDependencies.create(pip_packages=['azureml-sdk','scikit-learn==0.22.1','azureml-dataprep[pandas,fuse]>=1.1.14'])
env.python.conda_dependencies = cd

# Register environment to re-use later
env.register(workspace=ws)

print("Creating estimator")
script_params = {
    # to mount files referenced by mnist dataset
    '--data-folder': mnistFileDataset.as_named_input(datasetName).as_mount(),
    '--regularization': 0.5
}

est = Estimator(source_directory=scriptFolder,
              script_params=script_params,
              compute_target=compute_target,
              environment_definition=env,
              entry_script='train_logistic.py')

print("Running experiment")
run = exp.submit(config=est)
run

# specify show_output to True for a verbose log
run.wait_for_completion(show_output=False)

print("Model training metrics")
print(run.get_metrics())

print("Filenames associated with the run")
print(run.get_file_names())

# register model
print("Registering model")
model = run.register_model(model_name='sklearn_mnist', model_path='outputs/sklearn_mnist_model.pkl')
print(model.name, model.id, model.version, sep='\t')