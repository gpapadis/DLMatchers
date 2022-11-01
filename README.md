# DLMatchers
Deep Learning-based Entity Matching

Datasets are available [here](https://zenodo.org/record/7252010).

New baselines are available [here](https://github.com/gpapadis/thresholdBasedBaselines).

To create a Docker container for the main DL-based matching algorithms run

`sudo docker build -t mostmatchers mostmatchers`

To log into the container, use the following command:

`sudo docker run -it --entrypoint=/bin/bash mostmatchers`

To use the GPUs of the underlying infrastructure, [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian) should be installed and the flag

`--gpus all`

should be added to the command that initiates the Docker container.

More details are provided [here](https://github.com/gpapadis/DLMatchers/tree/main/dockers/mostmatchers).

To clean up all disk space occupied by Docker (after many experimentations), use the following commands:
* `sudo docker system prune -a`
* `sudo docker volume prune`
