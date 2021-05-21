import pickle

from ruamel import yaml

from pandas.tests.io.test_gbq import bigquery

def load_config_file(yaml_path):
    """
    Load the config yaml file as dictionary
    Args:
        config_fpath: filepath of yaml config file
    Returns:
        dictionary of loaded config
    """
    with open(yaml_path) as yaml_file:
        yaml_loaded = yaml.load(yaml_file, Loader=yaml.SafeLoader)
    return yaml_loaded

def export_embeddings_to_file(embeddings, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(embeddings, f)


def create_bq_client():
    credentials, your_project_id = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    bqclient = bigquery.Client(credentials=credentials, project=your_project_id,)
    bqstorageclient = bigquery_storage.BigQueryReadClient(credentials.credentials)
    return bqclient, bqstorageclient


def download_data_from_bq(table_name):
    bqclient, bqstorageclient = create_bq_client()
    query_string = f"SELECT * FROM `{table_name}`"
    dataframe = (bqclient.query(query_string).result().to_dataframe(bqstorage_client=bqstorageclient))
    return dataframe