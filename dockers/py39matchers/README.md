This Dockerfile runs [GNEM](https://www.cs.sjtu.edu.cn/~shen-yy/TheWebCon_2021_paper_3002.pdf).

To do so, build the Docker image with:

`sudo docker build -t py39matchers py39matchers`

and then log into the Docker container with:

`sudo docker run -it --entrypoint=/bin/bash --gpus all py39matchers`

and activate the corresponding conda environment with:

`conda activate p39`

Finally, use the commands in [GNEM's repository](https://github.com/ChenRunjin/GNEM).
