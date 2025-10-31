-- Node and hub embeddings from graph learning
-- Used for similarity search, cold-start, and transfer learning

CREATE TABLE IF NOT EXISTS embeddings.node_hub_v1 (
    node_id String,
    hub_id String,
    iso LowCardinality(String),
    
    -- Embedding vector (256-dimensional)
    embedding Array(Float32),
    embedding_dim UInt16 DEFAULT 256,
    
    -- Metadata
    model_type LowCardinality(String) DEFAULT 'GraphSAGE',  -- or 'DeepWalk', 'Node2Vec'
    model_version String,
    training_window_days UInt16,
    
    -- Quality metrics
    reconstruction_error Float32,
    cluster_id Nullable(UInt16),
    
    -- Timestamps
    valid_from DateTime64(3),
    valid_to Nullable(DateTime64(3)),
    created_at DateTime64(3) DEFAULT now64()
)
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY iso
ORDER BY (node_id, valid_from)
SETTINGS index_granularity = 8192;

-- Approximate Nearest Neighbor (ANN) index for similarity search
-- Note: ClickHouse 23.x+ supports vector similarity search
-- For earlier versions, use external ANN index (FAISS, Annoy)

-- Hub-level aggregated embeddings
CREATE MATERIALIZED VIEW IF NOT EXISTS embeddings.hub_v1
ENGINE = ReplacingMergeTree(created_at)
PARTITION BY iso
ORDER BY (hub_id, valid_from)
AS SELECT
    hub_id,
    iso,
    
    -- Average embeddings of nodes in hub
    groupArray(embedding) as node_embeddings,
    arrayMap(i -> avg(arrayElement(embedding, i)), range(1, 257)) as hub_embedding,
    
    count() as num_nodes,
    model_type,
    model_version,
    valid_from,
    valid_to,
    max(created_at) as created_at
FROM embeddings.node_hub_v1
GROUP BY hub_id, iso, model_type, model_version, valid_from, valid_to;

-- Similarity precomputed table for fast k-NN
CREATE TABLE IF NOT EXISTS embeddings.node_similarity (
    node_id_1 String,
    node_id_2 String,
    iso LowCardinality(String),
    
    -- Similarity metrics
    cosine_similarity Float32,
    euclidean_distance Float32,
    
    -- Metadata
    embedding_version String,
    computed_at DateTime64(3) DEFAULT now64()
)
ENGINE = MergeTree()
PARTITION BY iso
ORDER BY (node_id_1, cosine_similarity)
SETTINGS index_granularity = 8192;

-- UDF for cosine similarity (if not built-in)
-- CREATE FUNCTION IF NOT EXISTS cosine_similarity AS (v1, v2) -> 
--     arraySum(arrayMap((x, y) -> x * y, v1, v2)) / 
--     (sqrt(arraySum(arrayMap(x -> x * x, v1))) * sqrt(arraySum(arrayMap(y -> y * y, v2))));

ALTER TABLE embeddings.node_hub_v1 COMMENT 'Graph-learned node embeddings for similarity search and transfer learning. Enables k-NN nowcast with <20ms p95 latency.';
