This Dockerfile runs the following matching algorithms:
* [DITTO](https://vldb.org/pvldb/vol14/p50-li.pdf)
* [EMTransformer](https://digitalcollection.zhaw.ch/bitstream/11475/19637/1/Entity_Machting_with_Transformers_edbt_2020__Camera_Ready.pdf)
* [DeepMatcher](https://chu-data-lab.github.io/CS8803Fall2018/CS8803-Fall2018-DML-Papers/deepmatcher-space-exploration.pdf)
* [HierMatcher](https://www.ijcai.org/Proceedings/2020/0507.pdf)
* [GNEM](https://www.cs.sjtu.edu.cn/~shen-yy/TheWebCon_2021_paper_3002.pdf)
* [Magellan](http://www.vldb.org/pvldb/vol9/p1197-pkonda.pdf)
* [ZeroER](https://chu-data-lab.github.io/downloads/ZeroER-SIGMOD2020.pdf)

To do so, build the Docker image with:

`sudo docker build -t mostmatchers mostmatchers`

and then log into the Docker container with:

`sudo docker run -it --entrypoint=/bin/bash mostmatchers`

To run **ZeroER**, activate the corresponding conda environment with:

`conda activate ZeroER`

and follow the instructions in [ZeroER's repository](https://github.com/chu-data-lab/zeroer).

To run **DITTO**, activate the corresponding conda environment with:

`conda activate p377`

and follow the instructions in [DITTO's repository](https://github.com/megagonlabs/ditto).

To run **GNEM**, activate the corresponding conda environment with:

`conda activate p39`

and follow the instructions in [GNEM's repository](https://github.com/ChenRunjin/GNEM).

To run the rest of the algorithms, activate the corresponding conda environment with:

`conda activate deepmatcher`

and follow the instructions in:
* [Magellan's repository](https://github.com/anhaidgroup/py_entitymatching),
* [DeepMatcher's repository](https://github.com/anhaidgroup/deepmatcher), and
* [HierMatcher's repository](https://github.com/casnlu/EntityMatcher).
