import numpy as np
import tensorflow as tf

from deep_recommender_embeddings.src.preprocessing import clean_text


def get_tf_lookup_table_for_property(es_data_hits, field, clean=False):
    if clean:
        data_dict = {f['sort'][0]:clean_text(f['_source'][field]) for f in es_data_hits if field in f['_source']}
    else:
        data_dict = {f['sort'][0]:f['_source'][field] for f in es_data_hits if field in f['_source']}

    initializer = tf.lookup.KeyValueTensorInitializer(keys=list(data_dict.keys()), values=list(data_dict.values()), key_dtype=tf.string, value_dtype=tf.string)
    lookup_table = tf.lookup.StaticHashTable(initializer, default_value="nill")
    print(f"Tensorflow lookup table created for {field} with {len(data_dict.keys())} entries.")
    unique_values = np.array(list(set(data_dict.values())))
    return lookup_table,  unique_values


def get_tf_lookup_for_dict(data_dict):
    if type(list(data_dict.values())[0]) is list:
        data_dict = {key:np.array(value) for key, value in data_dict.items()}
    vec_shape = list(data_dict.values())[0].shape
    lookup_table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.string, value_dtype=tf.float32, empty_key="<EMPTY_SENTINEL>", deleted_key="<DELETE_SENTINEL>", default_value=tf.convert_to_tensor(np.zeros(shape=vec_shape), dtype=tf.float32))
    lookup_table.insert(list(data_dict.keys()), list(data_dict.values()))
    print(f"Tensorflow lookup table created for vector dim={vec_shape} with {len(data_dict.keys())} entries.")
    return lookup_table