# DLMatchers
This repository contains pointers to all code and data used in our publication on [A Critical Re-evaluation of Benchmark Datasets for (Deep) Learning-Based Matching Algorithms](https://arxiv.org/abs/2307.01231).

All datasets are available [here](https://zenodo.org/record/8164151).

The code that was used for generating the new benchmark datasets is available [here](https://github.com/gpapadis/DLMatchers/tree/main/DeepBlocker4NewDatasets). The input data to these scripts can be found [here](https://zenodo.org/record/7460624).

The implementation of the non-neural, linear supervised matching algorithms is available [here](https://github.com/gpapadis/thresholdBasedBaselines).

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
