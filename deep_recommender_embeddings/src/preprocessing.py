import re
import string
from glob import glob

import nltk
import pandas as pd
import numpy as np

from deep_recommender_embeddings.src.elasticsearch_utils import print_item

MAX_JOURNEY = 50
MIN_JOURNEY = 2

nltk.download('stopwords')
nltk.download('punkt')
nltk_stopwords = nltk.corpus.stopwords.words('english')

def load_data(d_path, stop_after_n_files=50, date_field='visit_start_datetime'):
    d_paths = glob(d_path)
    dateparse = lambda x: pd.to_datetime(x)
    df = []
    for i, f_path in enumerate(d_paths):
        if i == stop_after_n_files:
            break
        df.append(pd.read_csv(f_path, parse_dates=[date_field],  date_parser=dateparse))
    df = pd.concat(df)
    return df


def uri_from_url(url):

    url = url.replace(':::', '/').replace('::', '/').replace(':', '/').replace('/av', '')

    if url.rfind('/newsround') > 0:
        url = url[url.rfind('/newsround'):]
        url = truncate_after_asset_number(url)
    elif url.rfind('/news') > 0:
        url = url[url.rfind('/news'):]
        if url.find('/localnews') > 0:
            url = truncate_after_asset_number(url, '7')
        else:
            url = truncate_after_asset_number(url)
    elif url.rfind('/sport') > 0:
        url = url[url.rfind('/sport'):]
        url = truncate_after_asset_number(url)
    elif url.rfind('/culture') > 0:
        url = url[url.rfind('/culture'):]
        url = truncate_after_asset_number(url)
    elif url.rfind('/programmes') > 0:
        url = url[url.rfind('/programmes'):]
        url = truncate_after_programme_id(url)
    elif url.rfind('/reel/video') > 0:
        url = url[url.rfind('/reel/video'):]
        url = truncate_after_programme_id(url)
    elif url.rfind('/food') > 0:
        url = url[url.rfind('/food'):]
    elif url.rfind('/ideas') > 0:
        url = url[url.rfind('/ideas'):]
        url = truncate_after_programme_id(url)
    elif url.rfind('/bbcthree') > 0:
        url = url[url.rfind('/bbcthree'):]
    else:
        url = None

    return url


def clean_data(df, columns=['audience_id', 'visit_start_datetime', 'url'], uri_prefix=None):
    df = df[columns]
    df = df.dropna(how='any')
    if 'url' in df.columns:
        df['uri'] = df.url.apply(uri_from_url)
        df = df.dropna(how='any')
        df.drop_duplicates(['uri', 'audience_id'], inplace=True)
        df = df.drop(columns='url')
    if uri_prefix:
        df = df[df.uri.str.startswith(uri_prefix)]
    return df


def get_pairs(journey, min_len=MIN_JOURNEY, max_len=MAX_JOURNEY, window=7):
    '''

    Args:
        journey (pd.Dataframe):
        min_len (int): filter out journeys lower than this number
        max_len (int): filter out journeys higher than this number

    Returns:

    '''
    if len(journey) <= max_len and len(journey) >=min_len:
        distance_range = np.arange(1, window)
        uris = list(journey.uri)
        pairs = []
        for dist in distance_range:
            pairs = pairs + list(zip(uris, uris[dist:]))
        return [{"item_a": p[0], "item_b": p[1]} for p in pairs]
    else:
        return None

def get_pairs_location(journey, min_len=MIN_JOURNEY, max_len=MAX_JOURNEY, window=7):
    '''

    Args:
        journey (pd.Dataframe):
        min_len (int): filter out journeys lower than this number
        max_len (int): filter out journeys higher than this number

    Returns:

    '''
    if len(journey) <= max_len and len(journey) >=min_len:
        distance_range = np.arange(1, window)
        uris = list(journey.uri)
        locations = list(journey.geo_city_site_visited)
        pairs = []
        for dist in distance_range:
            pairs = pairs + list(zip(uris, locations, uris[dist:]))

        return [{"user_item": p[0], "user_location": p[1], "candidate_item": p[2]} for p in pairs]
    else:
        return None


def generate_model_recs(nn_engine, uri2id, id2uri, uri2vector, es_client, es_index, k, query_uri=None, query_vector=None):
    if query_vector is None:
        query_vector = uri2vector[query_uri]
    rec_ids = nn_engine.get_nns_by_vector(query_vector, k)
    rec_uris = [id2uri[r] for r in rec_ids]
    print('Generating recommendations for:')
    if query_uri:
        print_item(query_uri, es_client, es_index)
    print('-------------------')
    print('-------------------')
    return rec_uris


def clean_text(text):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text = text.translate(table)
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [token for token in tokens if not token in nltk_stopwords]
    return ' '.join(tokens)


def truncate_after_asset_number(url, n='8'):
    try:
        url = url[:re.search('\(|\)|\d{' + n + '}', url).end()]
        return url
    except AttributeError:
        #print(f'Warning: {url} has no asset number.')
        return None


def truncate_after_programme_id(url):
    p0 = url.rfind('p0')
    if p0 > 0:
        return url[:p0 + 8]
    else:
        return None


def filter_logs(logs, min_date, max_date, min_mentions, unique_item_ids, date_field='visit_start_datetime'):
    print(f'Original interactions: {len(logs)}')
    logs = logs[logs[date_field].between(min_date, max_date)]
    print(f'Date filtered interactions: {len(logs)}')

    logs = logs.groupby('uri').filter(lambda x: len(x) >= min_mentions)
    print(f'Minimum mention filtered interactions  : {len(logs)}')

    logs = logs[logs['uri'].isin(unique_item_ids)]
    print(f'Interactions filtered for items existing in ES : {len(logs)}')

    return logs


def get_item_pairs_from_journeys(journeys, window, date_field='visit_start_datetime', audience_id='audience_id',
                                 pair_function=get_pairs):
    journeys = journeys.sort_values(by=date_field).groupby(audience_id).apply(lambda x:  pair_function(x, window=window))
    journeys.dropna(inplace=True)
    all_pairs = [a for b in journeys.tolist() for a in b]
    print(f"Total number of item pairs: {len(all_pairs)}")
    return all_pairs