import datetime

import dateutil.parser
import elasticsearch
import numpy as np


def get_es_instance(es_host: str, es_port: str, es_auth=None):
    """Return an elasticsearch connection instance corresponding to the connection parameters given"""
    es_host_config = {"host": es_host, "port": es_port}
    es = elasticsearch.Elasticsearch(hosts=[es_host_config], http_auth=es_auth)
    if not es.ping():
        raise ConnectionError(f'Connection to ES failed with config: {es_host_config, es_auth}')
    return es


def get_item_info(asset_uri, es_client, es_index):
    query = {"query": {"match": {"assetUri": asset_uri}}}
    results = es_client.search(index=es_index, body=query)['hits']['hits']
    if len(results) == 0 or results[0]['_source']['assetUri'] != asset_uri:
        raise ValueError(f'{asset_uri} not found in elasticsearch index.')
    return results[0]


def get_data_from_es(es_client, fields, max_records, page_size, prediction_date="2020-11-02T00:00:00", max_days=90, es_index='index'):
    p_date = dateutil.parser.parse(prediction_date)
    days = datetime.timedelta(days=max_days)
    filter_date = p_date - days

    num_pages = int(np.ceil(max_records / page_size))
    hits = []
    for i in range(num_pages):
        print(i)
        if i < 1:
            q = {"query": {"bool": {"must": {"range": {"lastUpdated":{"gte": filter_date.isoformat()}}}}},
                 "_source": fields,
                 "size": page_size,
                 "sort": [{"assetUri": "asc"}]}
        else:

            search_after = res['hits']['hits'][-1]['sort']
            q = {"query": {"bool": {"must": {"range": {"lastUpdated":{"gte": filter_date.isoformat()}}}}},
                 "_source": fields,
                 "size": page_size,
                 "sort": [{"assetUri": "asc"}],
                 "search_after": search_after}

        res = es_client.search(index=es_index, body=q)
        hits = hits + res['hits']['hits']
        if len(res['hits']['hits']) == 0:
            break

    return hits


def print_item(query_uri, es_client, es_index):
    try:
        meta = get_item_info(query_uri, es_client, es_index)
        summary = meta["_source"]["summary"]
    except ValueError:
        summary = "NA"

    print(f'{query_uri} \n {summary}')