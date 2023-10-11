"""
 FAISS-based index components for dense retriever
 from 
 https://github.com/facebookresearch/DPR/blob/main/dense_retriever.py
 https://github.com/facebookresearch/DPR/blob/main/dpr/indexer/faiss_indexers.py
"""

import logging
from typing import List, Tuple
import os
import pickle
import time

import numpy as np
import faiss
from torch import nn
import torch

logger = logging.getLogger()


class DenseIndexer(object):
  def __init__(self, buffer_size: int = 50000):
    self.buffer_size = buffer_size
    self.index_id_to_db_id = []
    self.index = None

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


class DenseRetriever(object):
  def __init__(self, question_encoder: nn.Module, batch_size: int):
    self.question_encoder = question_encoder
    self.batch_size = batch_size
    self.selector = None


class LocalFaissRetriever(DenseRetriever):
  """
  Does passage retrieving over the provided index and question encoder
  """

  def __init__(
      self,
      question_encoder: nn.Module,
      batch_size: int,
      index: DenseIndexer,
  ):
    super().__init__(question_encoder, batch_size)
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
    self.index = None
    return results
