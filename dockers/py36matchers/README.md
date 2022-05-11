This Dockerfile runs [ZeroER](https://chu-data-lab.github.io/downloads/ZeroER-SIGMOD2020.pdf)

To do so, build the Docker image with:

`sudo docker build -t py36matchers py36matchers`

and then log into the Docker container with:

`sudo docker run -it --entrypoint=/bin/bash py36matchers`

and activate the corresponding conda environment with:

`conda activate ZeroER`

Finally, use the commands in [ZeroER's repository](https://github.com/chu-data-lab/zeroer).

Note that ZeroER is not a deep learning-based matching algorithm, but is included as a baseline method.
