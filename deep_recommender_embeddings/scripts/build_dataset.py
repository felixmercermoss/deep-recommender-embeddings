import os
import datetime

import pickle
import sys

import tensorflow as tf
import pandas as pd
import numpy as np
from deep_recommender_embeddings.src.ares import request_asset_from_ares
from deep_recommender_embeddings.src.elasticsearch_utils import get_es_instance, get_data_from_es, print_item
from deep_recommender_embeddings.src.image_embeddings import generate_image_embeddings
from deep_recommender_embeddings.src.models import ItemSimilarityModel
from deep_recommender_embeddings.src.plotting import plot_metric
from deep_recommender_embeddings.src.tf_utils import get_tf_lookup_table_for_property, get_tf_lookup_for_dict
from deep_recommender_embeddings.src.inference import get_dict_of_embeddings, build_annoy_index
from deep_recommender_embeddings.src.io import export_embeddings_to_file, load_config_file
from src.preprocessing import load_data, filter_logs


def main(config):

    logs = load_data(d_path=config['user_logs_path'], stop_after_n_files=None)
    logs = filter_logs(logs)
    journeys = logs.sort_values(by='visit_start_datetime').groupby(config['group_by_field'])







if __name__ == "__main__":
    config_fpath = sys.argv[0]
    config = load_config_file()
    main(config)