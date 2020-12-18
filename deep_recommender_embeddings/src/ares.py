import datetime
import os
import logging

import requests
import json

dev_cert = '/Users/mercef02/certificates/webUI/human-evaluation-tool.sport.datalab.api.bbci.co.uk.crt.pem'
dev_key = '/Users/mercef02/certificates/webUI/human-evaluation-tool.sport.datalab.api.bbci.co.uk.key.pem'

my_dev_cert='/Users/mercef02/certificates/key_and_cert/bbc_dev.crt.pem'
my_dev_key='/Users/mercef02/certificates/key_and_cert/bbc_dev.key.pem'

logger = logging.getLogger()

def get_date_from_timestamp(timestamp):
    if (timestamp / 1e11) > 1:
        timestamp = timestamp / 1000.
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%S+00:00')


def get_uris_from_cps_response(data):
    uris = [uri['assetUri'] for uri in data.get('results', [])]
    return uris

def get_value_or_none_from_asset(asset, keys):
    """
    Iteratively goes through the keys fields to get the specified value
    Example:
        {'promo': {'headlines' : {'headline': 'Test', 'shortHeadline': 'Short Test'}}}
        keys of ['promo', 'headlines', 'shortHeadline'] would return 'Short Test'
    Args:
        asset (dict): dict to retrieve value from
        keys (iterable): iterable of keys to use to get value
    Returns:
        value if exists else None
    """
    value = asset
    for key in keys:
        value = value.get(key, {})
    return value or None
def filter_ares_data(raw_document):
    """
    Takes a raw ARES response and returns a flattened dict containing just the information needed to display an item on


    """
    filtered_ares_item = {
        'image_url': get_value_or_none_from_asset(raw_document, ['promo', 'indexImage', 'href']),
        'asset_uri': get_value_or_none_from_asset(raw_document, ['promo', 'locators', 'assetUri']),
        'headline':  get_value_or_none_from_asset(raw_document, ['promo', 'headlines', 'headline']),
        'lastPublished': get_date_from_timestamp(
            get_value_or_none_from_asset(raw_document, ['metadata', 'lastPublished'])),
        'summary': get_value_or_none_from_asset(raw_document, ['promo', 'summary']),
        'first_created': get_value_or_none_from_asset(raw_document, ['metadata', 'firstCreated']),
        'article_id': get_value_or_none_from_asset(raw_document, ['metadata', 'locators', 'assetUri']).split('/')[-1]
    }
    return filtered_ares_item


def request_asset_from_ares(asset_uri='/news/uk-wales-north-west-wales-19795571',
                            ares_endpoint='https://ares-api.api.bbci.co.uk/api/asset/',
                            dev_cert_path=dev_cert, dev_key_path=dev_key, service='sfv'):
    """
    Makes call to ARES
    Input is the section of a url after 'bbc.co.uk/' on a live article on bbc website.

    Some example article URIs:
    assetURL = '/news/business-41649498'
    assetURL = '/news/uk-wales-north-west-wales-19795571'
    assetURL = '/sport/football/41656917'
    assetURL = '/sport/41723971'

    Args:
        asset_uri (str): section of a url after 'bbc.co.uk' on a live article on bbc website.
        ares_endpoint (str):Ares endpoint used tot fetchmetadata

    Returns:

    """

    headers = {'X-Candy-Platform': 'EnhancedMobile', 'X-Candy-Audience': 'International', 'Accept': 'application/json'}
    asset_uri = asset_uri[1:] if asset_uri.startswith("/") else asset_uri
    query = os.path.join(ares_endpoint, asset_uri)

    if not dev_cert_path:
        dev_cert_path = os.getenv('DEV_CERT_PATH')
        dev_key_path = os.getenv('DEV_KEY_PATH')

    if not dev_cert_path:
        logger.warning('No dev cert path found')
        return None

    cert = (dev_cert_path, dev_key_path)
    print(query)
    r = requests.get(query, headers=headers, cert=cert, verify=False)

    if r.status_code == 404 or r.status_code == 202:
        try:
            asset_uri_comp = asset_uri.split("/")
            asset_uri = "/".join(asset_uri_comp[0:1] + ["av"] + asset_uri_comp[1:])
            query = os.path.join(ares_endpoint, asset_uri)
            print(query)
            r = requests.get(query, headers=headers, cert=cert, verify=False)
            data = r.json()
            return filter_ares_data(data)
        except json.decoder.JSONDecodeError:
            logger.warning(f"Article ({asset_uri})no longer exists. Moving on...")
            return None

    if not r.status_code == 200:
        logger.warning(f'Something went wrong with ARES request {query}, returned: {r.status_code}')
        logger.warning(f"The document {asset_uri} may no longer exist.")
        logger.warning("Moving on...")
        return None
    else:
        try:
            data = r.json()
            return filter_ares_data(data)
        except json.decoder.JSONDecodeError:
            logger.warning(f"Article ({asset_uri})no longer exists. Moving on...")
            return None

