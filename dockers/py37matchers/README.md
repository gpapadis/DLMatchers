This Dockerfile runs [DITTO](https://vldb.org/pvldb/vol14/p50-li.pdf)

To do so, after building the Docker image with:

`sudo docker build -t py3Xmatchers py37matchers`

log into the Docker container with:

`sudo docker run -it --entrypoint=/bin/bash --gpus all py37matchers`

and activate the corresponding conda environment with:

`conda activate p377`

Then, use the commands in [DITTO's repository](https://github.com/megagonlabs/ditto).
