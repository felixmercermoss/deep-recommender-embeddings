import tensorflow as tf

from annoy import AnnoyIndex

from deep_recommender_embeddings.src.io import export_embeddings_to_file


def build_annoy_index(embeddings):
    embedding_dimension = len(list(embeddings.values())[0])
    annoy_index = AnnoyIndex(embedding_dimension, "angular")
    id_to_uri = {}
    uri_to_id = {}
    i=0
    for uri, embedding in embeddings.items():
        annoy_index.add_item(i, embedding)
        id_to_uri[i] = uri
        uri_to_id[uri] =  i
        i +=1

    # Build a 10-tree ANN index.
    annoy_index.build(100)
    return annoy_index, id_to_uri, uri_to_id


def get_dict_of_embeddings(trained_model, unique_item_ids, save_fpath=None):
    uri_to_embedding = {}
    items_tf = tf.data.Dataset.from_tensor_slices(unique_item_ids)
    item_embeddings = items_tf.batch(1).enumerate().map(
        lambda i, item_id: {'uri': item_id[0], 'id': i, 'embedding': trained_model.item_model(item_id)[0]})
    for item_embedding in item_embeddings.as_numpy_iterator():
        uri_to_embedding[str(item_embedding['uri'], 'utf-8')] = item_embedding['embedding']

    if save_fpath:
        export_embeddings_to_file(item_embeddings, save_fpath)

    return uri_to_embedding