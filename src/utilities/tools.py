def partition_batch(ls, size):
    """
    Partitions a list into buckets of given maximum length.
    """
    i = 0
    partitioned_lists = []
    while i < len(ls):
        partitioned_lists.append(ls[i: i+size])
        i += size
    return partitioned_lists
