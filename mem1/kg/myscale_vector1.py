import asyncio
from tqdm.asyncio import tqdm as tqdm_async
import numpy as np
from dataclasses import dataclass
from typing import Union,Any
from clickhouse_connect import get_client

from ..utils import logger
from ..base import (
    BaseVectorStorage,
)
from enum import Enum
class SortOrder(Enum):
    ASC = "ASC"
    DESC = "DESC"
class MyScaleConf:
    host: str
    port: int
    user: str
    password: str
    database: str
    fts_params: str

@dataclass
class MyScaleVectorStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    @staticmethod
    def create_collection_if_not_exist(
        client: any,conf:MyScaleConf, collection_name: str, dimension: int
    ):
        metric: str = "Cosine"
        client.command(f"CREATE DATABASE IF NOT EXISTS {conf.database}")
        fts_params = f"('{conf.fts_params}')" if conf.fts_params else ""
        sql = f"""
            CREATE TABLE IF NOT EXISTS {conf.database}.{collection_name}(
                id String,
                text String,
                vector Array(Float32),
                metadata JSON,
                CONSTRAINT cons_vec_len CHECK length(vector) = {dimension},
                VECTOR INDEX vidx vector TYPE DEFAULT('metric_type = {metric}'),
                INDEX text_idx text TYPE fts{fts_params}
            ) ENGINE = MergeTree ORDER BY id
        """
        client.command(sql)

    def __post_init__(self):
        conf=MyScaleConf(
            host=self.global_config["myscale_host"],
            port=self.global_config["myscale_port"],
            user=self.global_config["myscale_user"],
            password=self.global_config["myscale_password"],
            database=self.global_config["myscale_database"],
            fts_params=self.global_config["myscale_fts_params"])
        self._conf=conf
        self._client = get_client(
            host=conf.host, 
            port=conf.port, 
            username=conf.user, 
            password=conf.password, 
            database=conf.database)
        self._client.command("SET allow_experimental_object_type=1")
        MyScaleVectorStorage.create_collection_if_not_exist(
            self._client,
            self._conf,
            self.namespace,
            dimension=self.embedding_func.embedding_dim,
        )
       

    async def upsert(self, data: dict[str, dict]):
        """向向量数据库中插入数据"""
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "id": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["vector"] = embeddings[i]
        results = self._client.upsert(collection_name=self.namespace, data=list_data)
        return results

    async def index_done_callback(self):
        pass

    def _search(self,collection_name:str, dist: str, order: SortOrder, **kwargs: Any) -> list[dict]:
        top_k = kwargs.get("top_k", 4)
        score_threshold = float(kwargs.get("score_threshold") or 0.0)
        where_str = (
            f"WHERE dist < {1 - score_threshold}"
            if self._metric.upper() == "COSINE" and order == SortOrder.ASC and score_threshold > 0.0
            else ""
        )
        sql = f"""
            SELECT text, vector, metadata, {dist} as dist FROM {self._conf.database}.{collection_name}
            {where_str} ORDER BY dist {order.value} LIMIT {top_k}
        """
        try:
            return [
                {
                    'page_content':r["text"],
                    'vector':r["vector"],
                    'metadata':r["metadata"],
                }
                for r in self._client.query(sql).named_results()
            ]
        except Exception as e:
            logger.error(f"\033[91m\033[1m{type(e)}\033[0m \033[95m{str(e)}\033[0m")  # noqa:TRY401
            return []

    async def query(self, query: str, top_k=5) -> Union[dict, list[dict]]:
        embedding = await self.embedding_func([query])
        results = self._search(
            collection_name=self.namespace,
            data=embedding,
            limit=top_k,
            output_fields=list(self.meta_fields),
            search_params={"metric_type": "COSINE", "params": {"radius": 0.2}},
        )
        return [
            {**dp["entity"], "id": dp["id"], "distance": dp["distance"]}
            for dp in results[0]
        ]

