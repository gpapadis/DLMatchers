This Dockerfile runs the following matching algorithms:
* [ZeroER](https://chu-data-lab.github.io/downloads/ZeroER-SIGMOD2020.pdf)
* [DITTO](https://vldb.org/pvldb/vol14/p50-li.pdf)
* [JointBERT](http://vldb.org/pvldb/vol14/p1913-peeters.pdf)
* [EMTransformer](https://digitalcollection.zhaw.ch/bitstream/11475/19637/1/Entity_Machting_with_Transformers_edbt_2020__Camera_Ready.pdf)
* [DeepMatcher](https://chu-data-lab.github.io/CS8803Fall2018/CS8803-Fall2018-DML-Papers/deepmatcher-space-exploration.pdf)
* [HierMatcher](https://www.ijcai.org/Proceedings/2020/0507.pdf)
* [AutoML4EM](https://openproceedings.org/2021/conf/edbt/p260.pdf)
* [GNEM](https://www.cs.sjtu.edu.cn/~shen-yy/TheWebCon_2021_paper_3002.pdf)

To do so, build the Docker image with:

`sudo docker build -t allmatchers allmatchers`

and then log into the Docker container with:

`sudo docker run -it --entrypoint=/bin/bash allmatchers`

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

`conda activate jointbert`

and follow the instructions in [JointBERT's repository](https://github.com/wbsg-uni-mannheim/jointbert).
