const { Pool } = require('pg');

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

async function setupSchema() {
  await pool.query(`CREATE EXTENSION IF NOT EXISTS vector`);
  await pool.query(`
    CREATE TABLE IF NOT EXISTS acs_chunks (
      id SERIAL PRIMARY KEY,
      content TEXT NOT NULL,
      category TEXT,
      page_start INT,
      page_end INT,
      embedding vector(1536)
    )
  `);
  console.log('Schema ready');
}

async function buildIndex() {
  await pool.query('DROP INDEX IF EXISTS acs_chunks_embedding_idx');
  await pool.query(`
    CREATE INDEX acs_chunks_embedding_idx
    ON acs_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
  `);
  console.log('HNSW index built');
}

async function insertChunk(content, category, pageStart, pageEnd, embedding) {
  await pool.query(
    `INSERT INTO acs_chunks (content, category, page_start, page_end, embedding)
     VALUES ($1, $2, $3, $4, $5)`,
    [content, category, pageStart, pageEnd, JSON.stringify(embedding)]
  );
}

async function similaritySearch(queryEmbedding, topK = 5, category = null) {
  const params = [JSON.stringify(queryEmbedding), topK];
  let sql = `
    SELECT content, category, page_start, page_end,
           1 - (embedding <=> $1::vector) AS score
    FROM acs_chunks
  `;
  if (category) {
    sql += ` WHERE category = $3`;
    params.push(category);
  }
  sql += ` ORDER BY embedding <=> $1::vector LIMIT $2`;
  const result = await pool.query(sql, params);
  return result.rows;
}

async function getChunkCount() {
  const result = await pool.query('SELECT COUNT(*) FROM acs_chunks');
  return parseInt(result.rows[0].count);
}

async function getCategories() {
  const result = await pool.query(
    'SELECT DISTINCT category, COUNT(*) as chunk_count FROM acs_chunks GROUP BY category ORDER BY category'
  );
  return result.rows;
}

module.exports = { pool, setupSchema, buildIndex, insertChunk, similaritySearch, getChunkCount, getCategories };
