# deep-recommender-embeddings
A repository to experiment with different deep recommender system architectures using Tensorflow Recommenders.


## Install dependencies

In the root of this repo, run:

```
pipenv install
```

To load the data needed, for the main notebook to work you will need to have an Elasticsearch instance running with
an index containing specially enriched ARES data (specifically, an index that contains image urls for each doc).
This can be built using the `FELIX-DEV` branch of  `datascapes-elasticsearch`. 

## Building models

The notebook "messy_notebook.ipynb" contains blocks which builds datasets, trains models, evaluates the models
and exports embeddings to saved artifacts.