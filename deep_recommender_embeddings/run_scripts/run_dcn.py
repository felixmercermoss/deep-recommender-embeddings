import numpy as np
import pandas as pd
import tensorflow as tf

# import tensorflow_datasets as tfds

from deep_recommender_embeddings.src.models import DCN
from deep_recommender_embeddings.src.elasticsearch_utils import (
    get_data_from_es,
    get_es_instance,
)
from deep_recommender_embeddings.src.tf_utils import get_tf_lookup_table_for_property
from deep_recommender_embeddings.src.preprocessing import (
    load_data,
    clean_data,
    filter_logs,
    get_pairs,
)


def get_item_data():
    _es_index = "sfv_rn_test"
    _es_host = "localhost"
    _es_port = "9200"
    es = get_es_instance(es_host=_es_host, es_port=_es_port)
    prediction_time = "2020-12-02T00:00:00"
    max_age_days = 90

    features = [
        "combinedBodySummaryHeadline",
        "tagsText",
        "articleCategoryName",
        "thumbnailUrl",
    ]
    hits = get_data_from_es(
        es, features, 30000, 5000, prediction_time, max_age_days, _es_index
    )

    body_lookup_table, _ = get_tf_lookup_table_for_property(
        hits, "combinedBodySummaryHeadline", clean=True
    )
    tags_lookup_table, _ = get_tf_lookup_table_for_property(hits, "tagsText")
    category_lookup_table, unique_categories = get_tf_lookup_table_for_property(
        hits, "articleCategoryName"
    )
    # thumbnail_lookup_table, _ = get_tf_lookup_table_for_property(hits, "thumbnailUrl")
    # im_vec_lookup_table = get_tf_lookup_for_dict(im_embeddings)
    unique_item_ids = [hit["sort"][0] for hit in hits]
    print(f"Unique item ids: {len(unique_item_ids)}")
    vocabularies = {}
    vocabularies["body"] = body_lookup_table
    vocabularies["item_id"] = unique_item_ids
    vocabularies["tags"] = tags_lookup_table
    vocabularies["category"] = category_lookup_table
    return vocabularies


def get_user_journey(unique_item_ids):
    DATA_PATH = "/Users/shengd02/Documents/data/week_compact/*"
    logs = load_data(d_path=DATA_PATH, stop_after_n_files=3)
    logs = clean_data(logs)

    min_date = pd.to_datetime("2020-08-21T00:00:00")
    max_date = pd.to_datetime("2020-08-23T11:59:59")
    min_mentions = 5
    train_logs = filter_logs(logs, min_date, max_date, min_mentions, unique_item_ids)
    train_logs = train_logs[np.isin(train_logs["uri"], unique_item_ids)]
    print(f"Content filtered interactions: {len(train_logs)}")

    train_logs = (
        train_logs.sort_values(by="visit_start_datetime")
        .groupby("audience_id")
        .apply(lambda x: get_pairs(x))
    )
    train_logs.dropna(inplace=True)

    train_all_pairs = [a for b in train_logs.tolist() for a in b]
    print(f"Total number of item pairs: {len(train_all_pairs)}")
    train_all_pairs = dict(pd.DataFrame(train_all_pairs))
    train_logs_tf = tf.data.Dataset.from_tensor_slices(train_all_pairs)

    tf.random.set_seed(42)
    training_shuffled = train_logs_tf.shuffle(
        buffer_size=100_000, seed=42, reshuffle_each_iteration=False
    )
    train_pc = 0.9
    sfv_train = training_shuffled.take(np.floor(len(training_shuffled) * train_pc))
    sfv_val = training_shuffled.skip(np.floor(len(training_shuffled) * train_pc))
    print(f"Number of train pairs: {len(sfv_train)}")
    print(f"Number of test pairs: {len(sfv_val)}")
    return sfv_train, sfv_val


if __name__ == "__main__":
    vocabularies = get_item_data()
    sfv_train, sfv_val = get_user_journey(vocabularies["item_id"])
    num_epochs = 2
    lr = 0.01
    cached_train = sfv_train.batch(1024).cache()
    cached_test = sfv_val.batch(1024).cache()
    model_one_layer = DCN(
        use_cross_layer=False, deep_layer_sizes=[64, 32, 32], vocabularies=vocabularies
    )
    model_one_layer.compile(optimizer=tf.keras.optimizers.Adagrad(lr))

    one_layer_history = model_one_layer.fit(
        cached_train,
        validation_data=cached_test,
        validation_freq=1,
        epochs=num_epochs,
        verbose=1,
    )
    #    callbacks=[tensorboard_callback])

    print(
        f'Train accuracy: {one_layer_history.history["factorized_top_k/top_100_categorical_accuracy"][-1]}'
    )
    print(
        f'Validation accuracy: {one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]}'
    )
