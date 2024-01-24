import weaviate.classes as wvc

properties = [
    wvc.Property(name="documentId", data_type=wvc.DataType.TEXT),
    wvc.Property(name="chunkId", data_type=wvc.DataType.INT),
    wvc.Property(name="totalChunks", data_type=wvc.DataType.INT),
    wvc.Property(name="text", data_type=wvc.DataType.TEXT),
]