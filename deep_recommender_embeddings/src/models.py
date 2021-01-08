import uuid

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_recommenders as tfrs

from PIL import Image as Image
from tensorflow.python.framework.errors_impl import UnknownError
from tensorflow.python.keras.initializers.initializers_v2 import VarianceScaling


class ItemEmbeddingModel(tf.keras.Model):
    def __init__(
        self,
        features=["item_id", "body"],
        feature_dims=[32, 32],
        unique_item_ids=None,
        item_body_lookup=None,
        item_tags_lookup=None,
        item_category_lookup=None,
        item_image_lookup=None,
        image_embedding_lookup_table=None,
    ):
        super().__init__()

        self.features = features
        self.IMAGE_SHAPE = (96, 96)

        for feature, embedding_dim in zip(self.features, feature_dims):
            # Define run_scripts for item_id embedding generation
            if "item_id" == feature:

                self.item_id_embedding = tf.keras.Sequential(
                    [
                        tf.keras.layers.experimental.preprocessing.StringLookup(
                            vocabulary=unique_item_ids,
                            mask_token=None,
                            name="item_id_string_lookup_layer",
                        ),
                        tf.keras.layers.Embedding(
                            input_dim=len(unique_item_ids) + 1,
                            output_dim=embedding_dim,
                            name="item_id_embedding_layer",
                        ),
                    ],
                    name="sequential_id_embedding",
                )

            # Define run_scripts for body embedding generation
            if "body" == feature:
                max_tokens = 10_000
                self.item_body_lookup_table = item_body_lookup
                self.body_embedding = tf.keras.Sequential(
                    [
                        tf.keras.layers.experimental.preprocessing.TextVectorization(
                            max_tokens=max_tokens, name="body_text_vectorisation_layer"
                        ),
                        tf.keras.layers.Embedding(
                            input_dim=max_tokens,
                            output_dim=embedding_dim,
                            mask_zero=True,
                            name="body_embedding_layer",
                        ),
                        tf.keras.layers.GlobalAveragePooling1D(
                            name="body_global_averaging_pooling_layer"
                        ),
                    ],
                    name="sequential_body_embedding",
                )

                # Define run_scripts for tags embedding generation
            if "tags" == feature:
                max_tokens = 10_000
                self.item_tags_lookup_table = item_tags_lookup
                self.tags_embedding = tf.keras.Sequential(
                    [
                        tf.keras.layers.experimental.preprocessing.TextVectorization(
                            max_tokens=max_tokens, name="tags_text_vectorisation_layer"
                        ),
                        tf.keras.layers.Embedding(
                            input_dim=max_tokens,
                            output_dim=embedding_dim,
                            mask_zero=True,
                            name="tags_embedding_layer",
                        ),
                        tf.keras.layers.GlobalAveragePooling1D(
                            name="tags_global_averaging_pooling_layer"
                        ),
                    ],
                    name="sequential_tags_embedding",
                )

            if "category" == feature:
                _, values = item_category_lookup.export()
                self.item_category_lookup_table = item_category_lookup
                unique_categories = np.unique(values.numpy())
                self.category_embedding = tf.keras.Sequential(
                    [
                        tf.keras.layers.experimental.preprocessing.StringLookup(
                            vocabulary=unique_categories,
                            mask_token=None,
                            name="category_string_lookup_layer",
                        ),
                        tf.keras.layers.Embedding(
                            input_dim=len(unique_categories) + 1,
                            output_dim=embedding_dim,
                            name="category_embedding_layer",
                        ),
                    ],
                    name="sequential_category_embedding",
                )

            if "image_embedding" == feature:
                self.image_embedding_lookup_table = image_embedding_lookup_table
                _, values = self.image_embedding_lookup_table.export()
                if embedding_dim:
                    self.image_embedding_model = tf.keras.Sequential(
                        [
                            tf.keras.Input(shape=values.numpy()[0].shape),
                            tf.keras.layers.Dense(
                                embedding_dim,
                                activation="relu",
                                kernel_initializer=VarianceScaling(),
                                bias_initializer=VarianceScaling(),
                                name=f"image_dense_layer_{str(uuid.uuid4())}",
                            ),
                        ]
                    )

                else:
                    self.image_embedding_model = tf.keras.Sequential(
                        [tf.keras.Input(shape=values.numpy()[0].shape)]
                    )

            if isinstance(embedding_dim, dict):
                if embedding_dim.get("format") == "image":
                    # IMAGE_MODEL_LINK ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
                    # IMAGE_MODEL_LINK = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
                    model_link = embedding_dim.get(
                        "link",
                        "https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/4",
                    )
                    print(f"Using pretrained IMAGE run_scripts: {model_link}")
                    self.item_image_lookup_table = item_image_lookup
                    self.image_embedding = tf.keras.Sequential(
                        [
                            hub.KerasLayer(
                                model_link,
                                input_shape=self.IMAGE_SHAPE + (3,),
                                trainable=False,
                            ),
                            # tf.keras.layers.Embedding(input_dim=len(unique_item_ids) + 1, output_dim=embedding_dim, name="image_embedding_layer"),
                            #  tf.keras.layers.GlobalAveragePooling1D(name="image_global_averaging_pooling_layer")
                        ]
                    )

                if embedding_dim.get("format") == "text":
                    model_link = embedding_dim.get(
                        "link", "https://tfhub.dev/google/nnlm-en-dim50/2"
                    )
                    print(f"Using pretrained TEXT run_scripts: {model_link}")
                    self.item_body_lookup_table = item_body_lookup
                    self.text_model_embedding = tf.keras.Sequential(
                        [
                            hub.KerasLayer(
                                model_link,
                                input_shape=[],
                                dtype=tf.string,
                                trainable=False,
                            )
                            # tf.keras.layers.Embedding(input_dim=MODEL_DIM, output_dim=embedding_dim, mask_zero=True, name="text_model_embedding_layer")
                            # tf.keras.layers.GlobalAveragePooling1D(name="text_model_global_averaging_pooling_layer")
                        ]
                    )

    def read_image(self, file_url):
        file_url_string = file_url.numpy().decode()
        if file_url_string == "nill":
            return (
                np.zeros(shape=(1, self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 3)) + 0.5
            )
        try:
            im = tf.keras.utils.get_file(
                file_url_string.replace("/", "_"), file_url_string
            )
            im = Image.open(im).resize(self.IMAGE_SHAPE)
            im = np.array(im) / 255.0
            try:
                im = im[:, :, :3]
            except (IndexError, UnknownError):
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            return im[np.newaxis, ...]
        except:
            return (
                np.zeros(shape=(1, self.IMAGE_SHAPE[0], self.IMAGE_SHAPE[1], 3)) + 0.5
            )

    def call(self, inputs):

        feature_embeddings = []

        if "item_id" in self.features:
            feature_embeddings.append(self.item_id_embedding(inputs))

        if "body" in self.features:
            body = self.item_body_lookup_table.lookup(inputs)
            feature_embeddings.append(self.body_embedding(body))

        if "tags" in self.features:
            tags = self.item_tags_lookup_table.lookup(inputs)
            feature_embeddings.append(self.tags_embedding(tags))

        if "category" in self.features:
            category = self.item_category_lookup_table.lookup(inputs)
            feature_embeddings.append(self.category_embedding(category))

        if "image_embedding" in self.features:
            image_vector = self.image_embedding_lookup_table.lookup(inputs)
            image_vector = self.image_embedding_model(image_vector)
            feature_embeddings.append(image_vector)

        if "image" in self.features:
            image_url = self.item_image_lookup_table.lookup(inputs)
            image = tf.map_fn(
                fn=lambda s: tf.py_function(self.read_image, inp=[s], Tout=tf.float64),
                elems=image_url,
                fn_output_signature=tf.float64,
            )
            image = tf.cast(tf.squeeze(image), tf.float32)
            feature_embeddings.append(self.image_embedding(image))

        if "text_model" in self.features:
            text = self.item_body_lookup_table.lookup(inputs)
            feature_embeddings.append(self.text_model_embedding(text))

        return tf.concat(feature_embeddings, axis=1)


class DeepItemModel(tf.keras.Model):
    def __init__(
        self,
        features=["item_id", "body"],
        feature_dims=[32, 32],
        unique_item_ids=None,
        item_body_lookup=None,
        item_tags_lookup=None,
        item_category_lookup=None,
        item_image_lookup=None,
        image_embedding_lookup_table=None,
        use_deep_layers=True,
        use_cross_layers=True,
        deep_layer_sizes=[32],
        cross_layer_projection_size=[32],
    ):
        """Model for encoding items.

        Args:
        deep_layer_sizes:
          A list of integers where the i-th entry represents the number of units
          the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = ItemEmbeddingModel(
            features=features,
            feature_dims=feature_dims,
            unique_item_ids=unique_item_ids,
            item_body_lookup=item_body_lookup,
            item_tags_lookup=item_tags_lookup,
            item_category_lookup=item_category_lookup,
            item_image_lookup=item_image_lookup,
            image_embedding_lookup_table=image_embedding_lookup_table,
        )

        # Then construct the layers.
        self.dcn_layers = tf.keras.Sequential(name="sequential_deep")

        if use_deep_layers:
            # Use the ReLU activation for all but the last layer.
            for i, layer_size in enumerate(deep_layer_sizes[:-1]):
                self.dcn_layers.add(
                    tf.keras.layers.Dense(
                        layer_size,
                        activation="relu",
                        kernel_initializer=VarianceScaling(),
                        bias_initializer=VarianceScaling(),
                        name=f"dense_layer_{i+1}_{str(uuid.uuid4())}",
                    )
                )
        if use_cross_layers:
            # Use the ReLU activation for all but the last layer.
            for i, projection_dim in enumerate(cross_layer_projection_size[:-1]):
                self.dcn_layers.add(
                    tfrs.layers.dcn.Cross(
                        projection_dim=projection_dim,
                        kernel_initializer="glorot_uniform",
                        name=f"cross_layer_{i+1}_{str(uuid.uuid4())}",
                    )
                )
        #         Batch normalization after the first layer
        #         if i == 0:
        #             self.dense_layers.add(tf.keras.layers.BatchNormalization())

        # No activation for the last layer.
        self.dcn_layers.add(
            tf.keras.layers.Dense(
                layer_size,
                kernel_initializer=VarianceScaling(),
                bias_initializer=VarianceScaling(),
                name=f"dense_layer_{len(deep_layer_sizes)}_{str(uuid.uuid4())}",
            )
        )

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dcn_layers(feature_embedding)


class ItemSimilarityModel(tfrs.Model):
    # We derive from a custom base class to help reduce boilerplate. Under the hood,
    # these are still plain Keras Models.

    def __init__(
        self,
        test_candidate_ids,
        features=["item_id", "body"],
        feature_dims=[32, 32],
        unique_item_ids=None,
        item_body_lookup=None,
        item_tags_lookup=None,
        item_category_lookup=None,
        item_image_lookup=None,
        image_embedding_lookup_table=None,
        layer_sizes=None,
        pretrained_text_model=None,
        pretrained_image_model=None,
        compute_metrics=True,
    ):
        super().__init__()
        self.compute_metrics = compute_metrics
        self.pretrained_text_model = pretrained_text_model
        self.pretrained_image_model = pretrained_image_model
        if layer_sizes:
            self.item_model = DeepItemModel(
                features=features,
                unique_item_ids=unique_item_ids,
                feature_dims=feature_dims,
                item_body_lookup=item_body_lookup,
                item_tags_lookup=item_tags_lookup,
                item_category_lookup=item_category_lookup,
                item_image_lookup=item_image_lookup,
                image_embedding_lookup_table=image_embedding_lookup_table,
                deep_layer_sizes=layer_sizes,
            )
        else:
            self.item_model = ItemEmbeddingModel(
                features=features,
                feature_dims=feature_dims,
                unique_item_ids=unique_item_ids,
                item_body_lookup=item_body_lookup,
                item_tags_lookup=item_tags_lookup,
                item_category_lookup=item_category_lookup,
                item_image_lookup=item_image_lookup,
                image_embedding_lookup_table=image_embedding_lookup_table,
            )

        # if isinstance(test_candidate_ids, np.ndarray):
        test_candidate_ids = tf.data.Dataset.from_tensor_slices(test_candidate_ids)

        self.task = tfrs.tasks.Retrieval(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=test_candidate_ids.batch(512).map(self.item_model)
            ),
        )

    def compute_loss(
        self, raw_features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        # Define how the loss is computed.

        item_a_embeddings = self.item_model(raw_features["item_a"])
        item_b_embeddings = self.item_model(raw_features["item_b"])

        return self.task(
            query_embeddings=item_a_embeddings,
            candidate_embeddings=item_b_embeddings,
            # candidate_ids=raw_features["item_b"],
            compute_metrics=self.compute_metrics,
        )


class DeepCrossItemModel(tf.keras.Model):
    def __init__(
        self,
        features=["item_id", "body"],
        feature_dims=[32, 32],
        unique_item_ids=None,
        item_body_lookup=None,
        item_tags_lookup=None,
        item_category_lookup=None,
        item_image_lookup=None,
        image_embedding_lookup_table=None,
        layer_sizes=[32],
        use_cross_layer=False,
        projection_dim=None,
    ):
        """Model for encoding items.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = ItemEmbeddingModel(
            features=features,
            feature_dims=feature_dims,
            unique_item_ids=unique_item_ids,
            item_body_lookup=item_body_lookup,
            item_tags_lookup=item_tags_lookup,
            item_category_lookup=item_category_lookup,
            item_image_lookup=item_image_lookup,
            image_embedding_lookup_table=image_embedding_lookup_table,
        )

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential(name="sequential_deep")

        if use_cross_layer:
            self.dense_layers.add(
                tfrs.layers.dcn.Cross(
                    projection_dim=projection_dim, kernel_initializer="glorot_uniform"
                )
            )

        # Use the ReLU activation for all but the last layer.
        for i, layer_size in enumerate(layer_sizes[:-1]):
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size,
                    activation="relu",
                    kernel_initializer=VarianceScaling(),
                    bias_initializer=VarianceScaling(),
                    name=f"dense_layer_{i + 1}_{str(uuid.uuid4())}",
                )
            )
        #         Batch normalization after the first layer
        #         if i == 0:
        #             self.dense_layers.add(tf.keras.layers.BatchNormalization())

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size,
                    kernel_initializer=VarianceScaling(),
                    bias_initializer=VarianceScaling(),
                    name=f"dense_layer_{len(layer_sizes)}_{str(uuid.uuid4())}",
                )
            )

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class DCN(tfrs.Model):
    def __init__(
        self,
        use_cross_layer,
        deep_layer_sizes,
        vocabularies,
        projection_dim=None,
        compute_metrics=True,
    ):
        super().__init__()

        self.embedding_dimension = [32, 32, 32, 32]

        str_features = ["body", "tags", "category"]
        id_features = ["item_id"]

        self.item_model = DeepCrossItemModel(
            features=str_features + id_features,
            feature_dims=self.embedding_dimension,
            unique_item_ids=vocabularies.get("item_id", None),
            item_body_lookup=vocabularies.get("body", None),
            item_tags_lookup=vocabularies.get("tags", None),
            item_category_lookup=vocabularies.get("category", None),
            item_image_lookup=vocabularies.get("image", None),
            image_embedding_lookup_table=vocabularies.get("image_embedding", None),
            use_cross_layer=use_cross_layer,
            projection_dim=projection_dim,
            layer_sizes=deep_layer_sizes,
        )

        self.compute_metrics = compute_metrics

        test_candidate_ids = tf.data.Dataset.from_tensor_slices(
            vocabularies.get("item_id", None)
        )
        self.task = tfrs.tasks.Retrieval(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=test_candidate_ids.batch(512).map(self.item_model)
            ),
        )

    def compute_loss(
        self, raw_features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        # Define how the loss is computed.

        item_a_embeddings = self.item_model(raw_features["item_a"])
        item_b_embeddings = self.item_model(raw_features["item_b"])

        return self.task(
            query_embeddings=item_a_embeddings,
            candidate_embeddings=item_b_embeddings,
            compute_metrics=self.compute_metrics,
        )
