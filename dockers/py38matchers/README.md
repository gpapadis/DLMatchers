This Dockerfile runs the following deep learning-based matchers:
* [JointBERT](http://vldb.org/pvldb/vol14/p1913-peeters.pdf)
* [EMTransformer](https://digitalcollection.zhaw.ch/bitstream/11475/19637/1/Entity_Machting_with_Transformers_edbt_2020__Camera_Ready.pdf)
* [DeepMatcher](https://chu-data-lab.github.io/CS8803Fall2018/CS8803-Fall2018-DML-Papers/deepmatcher-space-exploration.pdf)
* [HierMatcher](https://www.ijcai.org/Proceedings/2020/0507.pdf)
* [AutoML4EM](https://openproceedings.org/2021/conf/edbt/p260.pdf)

It also runs the machine learning-based matcher [Magellan](http://www.vldb.org/pvldb/vol9/p1197-pkonda.pdf).

To do so, build the Docker image with:

`sudo docker build -t py38matchers py38matchers`

and then log into the Docker container with:

`sudo docker run -it --entrypoint=/bin/bash --gpus all py38matchers`

and activate the corresponding conda environment with:

`conda activate jointbert`

Finally, use the commands in [JointBERT's repository](https://github.com/wbsg-uni-mannheim/jointbert).

The source-code for the rest of the methods is available here:
* [EMTransformer](https://github.com/brunnurs/entity-matching-transformer)
* [DeepMatcher](https://github.com/anhaidgroup/deepmatcher)
* [HierMatcher](https://github.com/casnlu/EntityMatcher)
* [AutoML4EM](https://github.com/softlab-unimore/automl-for-em)
* [Magellan](https://github.com/anhaidgroup/py_entitymatching)
