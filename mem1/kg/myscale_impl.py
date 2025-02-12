import logging
from typing import List, Optional, Tuple
import uuid
from clickhouse_connect import get_client

logger = logging.getLogger(__name__)

class MyScaleVectorStorage:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        database: str,
        table: str,
        vector_dim: int
    ):
        self.client = get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database
        )
        self.database = database
        self.table = table
        self.vector_dim = vector_dim

    def create_table(self):
        """创建带有向量列和元数据列的表"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table}
        (
            id String,
            embedding Array(Float32),
            metadata Map(String, String),
            document String,
            CONSTRAINT vector_len CHECK length(embedding) = {self.vector_dim}
        ) ENGINE = MergeTree()
        ORDER BY id
        """
        
        # 创建向量索引
        index_sql = f"""
        ALTER TABLE {self.table} 
        ADD VECTOR INDEX vec_idx embedding TYPE MSTG
        OPTIONS(
            metric_type='L2',
            max_patrol_steps=100
        )
        """
        
        try:
            self.client.execute(create_table_sql)
            self.client.execute(index_sql)
        except ClickhouseError as e:
            logger.error(f"创建表失败: {str(e)}")
            raise

    def insert(
        self,
        vectors: List[List[float]],
        metadatas: List[dict],
        documents: List[str]
    ) -> List[str]:
        """插入向量数据"""
        ids = [str(uuid.uuid4()) for _ in documents]
        data = [
            {
                "id": ids[i],
                "embedding": vectors[i],
                "metadata": metadatas[i],
                "document": documents[i]
            }
            for i in range(len(documents))
        ]
        
        insert_sql = f"""
        INSERT INTO {self.table} 
        (id, embedding, metadata, document)
        VALUES
        """
        try:
            self.client.execute(insert_sql, data)
            return ids
        except ClickhouseError as e:
            logger.error(f"插入数据失败: {str(e)}")
            raise

    def search(
        self,
        query_vector: List[float],
        limit: int = 10
    ) -> List[Tuple[str, float, str]]:
        """相似向量搜索"""
        search_sql = f"""
        SELECT 
            id, 
            L2Distance(embedding, {query_vector}) as distance,
            document
        FROM {self.table}
        ORDER BY distance ASC
        LIMIT {limit}
        """
        
        try:
            result = self.client.execute(search_sql)
            return [(row[0], row[1], row[2]) for row in result]
        except ClickhouseError as e:
            logger.error(f"搜索失败: {str(e)}")
            raise

    def delete(self, ids: List[str]):
        """按ID删除数据"""
        delete_sql = f"""
        ALTER TABLE {self.table}
        DELETE WHERE id IN ({','.join(['%s']*len(ids))})
        """
        try:
            self.client.execute(delete_sql, ids)
        except ClickhouseError as e:
            logger.error(f"删除失败: {str(e)}")
            raise

    def table_exists(self) -> bool:
        """检查表是否存在"""
        check_sql = f"""
        SELECT count()
        FROM system.tables
        WHERE database = '{self.database}' AND name = '{self.table}'
        """
        try:
            result = self.client.execute(check_sql)
            return result[