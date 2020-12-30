import pickle


def export_embeddings_to_file(embeddings, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(embeddings, f)