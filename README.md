# DLMatchers
Deep Learning-based Entity Matching

To create a Docker container for each version of Python, use the following command:

`sudo docker build -t py3Xmatchers py3Xmatchers`

To log into the container, use the following command:

`sudo docker run -it --entrypoint=/bin/bash py3Xmatchers`

To use the GPUs of the underlying infrastructure, [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian) should be installed and the flag

`--gpus all`

should be added to the command that initiates the Docker container.

To clean up all disk space occupied by Docker (after many experimentations), use the following commands:
* `sudo docker system prune -a`
* `sudo docker volume prune`
