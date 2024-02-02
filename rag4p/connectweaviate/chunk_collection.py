import weaviate.classes.config as wvc

properties = [
    wvc.Property(name="documentId",
                 data_type=wvc.DataType.TEXT,
                 vectorize_property_name=False,
                 skip_vectorization=True),
    wvc.Property(name="chunkId",
                 data_type=wvc.DataType.INT,
                 vectorize_property_name=False,
                 skip_vectorization=True),
    wvc.Property(name="totalChunks",
                 data_type=wvc.DataType.INT,
                 vectorize_property_name=False,
                 skip_vectorization=True),
    wvc.Property(name="text",
                 data_type=wvc.DataType.TEXT,
                 vectorize_property_name=False,
                 skip_vectorization=False),
]
