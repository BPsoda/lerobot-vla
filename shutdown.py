# Shut down the container by sending a request to the shutdown endpoint
import os
import requests
import subprocess


def shutdown():
    '''Make request to k8s and shutdown the container.'''
    name = os.environ.get('K8S_USER_NAME')
    password = os.environ.get('K8S_USER_PASSWORD')
    host_result = subprocess.run(['hostname'], capture_output=True, text=True)
    assert name and password, "Name or password no found, set them in your environment variables!"
    vmids = [host_result.stdout.strip()]
    url = "http://k8svmgr-main.devops.svc.cluster.local:8000/api/task_finished"
    data = {
        "name": name,
        "password": password,
        "vmids": vmids
    }
    response = requests.post(url, json=data)
    print(response.text)
    return response

if __name__ == "__main__":
    shutdown()