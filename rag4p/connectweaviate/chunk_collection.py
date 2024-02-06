import weaviate.classes.config as wvc


def weaviate_properties(additional_properties=None):

    if additional_properties is None:
        additional_properties = []

    common_properties = [
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

    return common_properties + additional_properties
