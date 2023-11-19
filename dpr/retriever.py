"""
FAISS-based index components for dense retriever
refactored from DPR codebase
https://github.com/facebookresearch/DPR/blob/main/dense_retriever.py
https://github.com/facebookresearch/DPR/blob/main/dpr/indexer/faiss_indexers.py
"""

import collections
import logging
from typing import List, Tuple
import os
import pickle
import time
import torch

import numpy as np
import faiss

logger = logging.getLogger()

BiEncoderPassage = collections.namedtuple(
    "BiEncoderPassage", ["text", "title"])


class DenseIndexer():
  """
  Load(deserialize) faiss index file or Save(serialize) faiss index object
  """

  def __init__(self, buffer_size: int = 50000):
    self.buffer_size = buffer_size
    self.index_id_to_db_id: List[int] = []
    self.index: faiss.Index = None

  def init_index(self, vector_sz: int):
    raise NotImplementedError

  def index_data(self, data: List[Tuple[object, np.array]]):
    raise NotImplementedError

  def get_index_name(self):
    raise NotImplementedError

  def search_knn(self, query_vectors: np.array, top_docs: int) -> List[Tuple[List[object], List[float]]]:
    raise NotImplementedError

  def serialize(self, file: str):
    logger.info("Serializing index to %s", file)

    if os.path.isdir(file):
      index_file = os.path.join(file, "index.dpr")
      meta_file = os.path.join(file, "index_meta.dpr")
    else:
      index_file = file + ".index.dpr"
      meta_file = file + ".index_meta.dpr"

    faiss.write_index(self.index, index_file)
    with open(meta_file, mode="wb") as f:
      pickle.dump(self.index_id_to_db_id, f)

  def get_files(self, path: str):
    if os.path.isdir(path):
      index_file = os.path.join(path, "index.dpr")
      meta_file = os.path.join(path, "index_meta.dpr")
    else:
      raise ValueError(f"Index path {path} is not a directory")
    return index_file, meta_file

  def index_exists(self, path: str) -> bool:
    index_file, meta_file = self.get_files(path)
    return os.path.isfile(index_file) and os.path.isfile(meta_file)

  def deserialize(self, path: str):
    logger.info("Loading index from %s", path)
    index_file, meta_file = self.get_files(path)

    self.index = faiss.read_index(index_file)
    logger.info("Loaded index of type %s and size %d",
                type(self.index), self.index.ntotal)

    with open(meta_file, "rb") as reader:
      self.index_id_to_db_id = pickle.load(reader)
    assert (
        len(self.index_id_to_db_id) == self.index.ntotal
    ), "Deserialized index_id_to_db_id should match faiss index size"

  def _update_id_mapping(self, db_ids: List) -> int:
    self.index_id_to_db_id.extend(db_ids)
    return len(self.index_id_to_db_id)


class DenseHNSWFlatIndexer(DenseIndexer):
  """
  Efficient index for retrieval. Note: default settings are for hugh accuracy but also high RAM usage
  """

  def __init__(
      self,
      buffer_size: float = 1e9,
      store_n: int = 512,
      ef_search: int = 128,
      ef_construction: int = 200,
  ):
    super().__init__()
    self.store_n = store_n
    self.ef_search = ef_search
    self.ef_construction = ef_construction
    self.phi = 0

  def init_index(self, vector_sz: int):
    # IndexHNSWFlat supports L2 similarity only
    # so we have to apply DOT -> L2 similairy space conversion with the help of an extra dimension
    index = faiss.IndexHNSWFlat(vector_sz + 1, self.store_n)
    index.hnsw.efSearch = self.ef_search
    index.hnsw.efConstruction = self.ef_construction
    self.index = index

  def index_data(self, data: List[Tuple[object, np.array]]):
    n = len(data)

    # max norm is required before putting all vectors in the index to convert inner product similarity to L2
    if self.phi > 0:
      raise RuntimeError(
          "DPR HNSWF index needs to index all data at once, results will be unpredictable otherwise."
      )
    phi = 0
    for i, item in enumerate(data):
      _, doc_vector = item[0:2]
      norms = (doc_vector ** 2).sum()
      phi = max(phi, norms)
    logger.info("HNSWF DotProduct -> L2 space phi=%s", phi)
    self.phi = phi

    # indexing in batches is beneficial for many faiss index types
    bs = int(self.buffer_size)
    for i in range(0, n, bs):
      db_ids = [t[0] for t in data[i: i + bs]]
      vectors = [np.reshape(t[1], (1, -1)) for t in data[i: i + bs]]

      norms = [(doc_vector ** 2).sum() for doc_vector in vectors]
      aux_dims = [np.sqrt(phi - norm) for norm in norms]
      hnsw_vectors = [np.hstack((doc_vector, aux_dims[i].reshape(-1, 1)))
                      for i, doc_vector in enumerate(vectors)]
      hnsw_vectors = np.concatenate(hnsw_vectors, axis=0)

      self._update_id_mapping(db_ids)
      self.index.add(hnsw_vectors)
      logger.info("data indexed %d", len(self.index_id_to_db_id))
    indexed_cnt = len(self.index_id_to_db_id)
    logger.info("Total data indexed %d", indexed_cnt)

  def search_knn(self, query_vectors: torch.Tensor, top_docs: int) -> List[Tuple[List[object], List[float]]]:
    logger.info("query_hnsw_vectors %s", query_vectors.shape)
    scores, indexes = self.index.search(query_vectors, top_docs)
    # convert to external ids
    db_ids = [[self.index_id_to_db_id[i] for i in query_top_idxs]
              for query_top_idxs in indexes]
    result = [(db_ids[i], scores[i]) for i in range(len(db_ids))]
    return result

  def deserialize(self, path: str):
    super().deserialize(path)
    # to trigger exception on subsequent indexing
    self.phi = 1

  def get_index_name(self):
    return "hnsw_index"


class LocalFaissRetriever():
  """
  Does passage retrieving over the provided index and question encoder
  """

  def __init__(
      self,
      index: DenseIndexer,
  ):
    super().__init__()
    self.index = index

  def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
    """
    Does the retrieval of the best matching passages given the query vectors batch
    :param query_vectors:
    :param top_docs:
    :return:
    """
    time0 = time.time()
    results = self.index.search_knn(query_vectors, top_docs)
    logger.info("index search time: %f sec.", time.time() - time0)
    return results
