from mem1.kg.myscale_vector import MyScaleVectorStorage, MyScaleVectorStorageConfig
#myscale_vector.py测试
if __name__ == "__main__":
    config = MyScaleVectorStorageConfig(
        host="192.168.195.29",
        port=11000,
        username="user1",
        database="test",
        password="password2",
        vector_dim=768,
        index_type="SCANN",
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