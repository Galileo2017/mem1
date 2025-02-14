import asyncio
import json
from dataclasses import dataclass
from typing import Dict, List, Union
import numpy as np
from tqdm.asyncio import tqdm as tqdm_async
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from clickhouse_driver import Client as ClickHouseClient
from clickhouse_driver.errors import NetworkError, SocketTimeoutError

from ..utils import logger
from ..base import (
    BaseVectorStorage,
)

@dataclass
class MyScaleVectorStorageConfig:
    host: str
    port: int
    username: str
    password: str
    database: str = "default"
    vector_dim: int = 768         # 向量维度
    index_type: str = "IVFSQ"       # 索引类型
    metric_type: str = "Cosine"   # 距离度量类型
    max_connections: int = 10     # 最大连接数

@dataclass
class MyScaleVectorStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2
    config: MyScaleVectorStorageConfig=None
    client: ClickHouseClient = None
    
    def __post_init__(self):
        self._init_client()
        self._verify_table_structure()

    def _init_client(self):
        """初始化MyScale连接"""
        self.client = ClickHouseClient(
            host=self.config.host,
            port=self.config.port,
            user=self.config.username,
            password=self.config.password,
            database=self.config.database,
            settings={'use_numpy': True}  # 启用numpy支持
        )
        
    def _verify_table_structure(self):
        """验证并创建表结构"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.namespace} (
            id String,
            content String,
            vector Array(Float32),
            metadata Map(String, String),
            created_at DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY id;
        """
        self.client.execute(create_table_sql)
        #创建索引
        create_index_sql = f"ALTER TABLE {self.namespace} ADD VECTOR INDEX vec_idx vector TYPE {self.config.index_type}('metric_type={self.config.metric_type}');"
        self.client.execute(create_index_sql)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((NetworkError, SocketTimeoutError))
    )
    async def upsert(self, data: Dict[str, dict]):
        """批量插入/更新向量数据"""
        if not data:
            return

        # 生成批量插入数据
        vectors = []
        contents = []
        ids = []
        metadata = []
        
        # 批量生成嵌入
        content_batches = [
            list(data.values())[i:i+self._max_batch_size] 
            for i in range(0, len(data), self._max_batch_size)
        ]
        
        for batch in tqdm_async(content_batches, desc="Generating embeddings"):
            embeddings = await self.embedding_func(
                [item["content"] for item in batch]
            )
            for item, vector in zip(batch, embeddings):
                ids.append(item["id"])
                contents.append(item["content"])
                vectors.append(vector.tolist())
                metadata.append(json.dumps(item.get("metadata", {})))

        # 使用ClickHouse的批量插入
        insert_sql = f"""
        INSERT INTO {self.namespace} (
            id, content, vector, metadata
        ) VALUES"""
        
        self.client.execute(
            insert_sql,
            [{
                "id": _id,
                "content": content,
                "vector": vector,
                "metadata": meta
            } for _id, content, vector, meta in zip(ids, contents, vectors, metadata)],
            types_check=True
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((NetworkError, SocketTimeoutError))
    )
    async def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """向量相似性搜索"""
        # 生成查询向量
        query_embedding = (await self.embedding_func([query]))[0].tolist()
        
        search_sql = f"""
        SELECT 
            id,
            content,
            metadata,
            distance(vector, {query_embedding}) AS score
        FROM {self.namespace}
        ORDER BY score DESC
        LIMIT {top_k}
        """
        
        result = self.client.execute(search_sql)
        return [{
            "id": row[0],
            "content": row[1],
            "metadata": json.loads(row[2]),
            "score": row[3]
        } for row in result]

    async def index_done_callback(self):
        """索引构建完成后的优化操作"""
        optimize_sql = f"""
        OPTIMIZE TABLE {self.namespace} 
        FINAL
        """
        self.client.execute(optimize_sql)
        logger.info(f"Optimized MyScale table {self.namespace}")

    async def delete_by_ids(self, ids: List[str]):
        """按ID批量删除"""
        delete_sql = f"""
        ALTER TABLE {self.namespace}
        DELETE WHERE id IN (%(ids)s)
        """
        self.client.execute(delete_sql, {"ids": ids})

# 使用示例
if __name__ == "__main__":
    config = MyScaleVectorStorageConfig(
        host="192.168.195.29",
        port=8126,
        username="default",
        database="test",
        password="",
        vector_dim=768,
        index_type="IVF_PQ",
        metric_type="Cosine"
    )
    
    storage = MyScaleVectorStorage(
        namespace="doc_vectors",
        global_config={"embedding_batch_num": 64},
        embedding_func=lambda x: np.random.rand(len(x), 768),  # 示例嵌入函数
        config=config
    )
    
    # 测试插入数据
    test_data = {
        f"doc_{i}": {
            "id": f"doc_{i}",
            "content": f"Document content {i}",
            "metadata": {"source": "test"}
        } for i in range(100)
    }
    asyncio.run(storage.upsert(test_data))
    
    # 测试查询
    results = asyncio.run(storage.query("test query", top_k=3))
    print("Top 3 results:", results)