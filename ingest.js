const { execSync } = require('child_process');
const fs = require('fs');
const OpenAI = require('openai');
const { setupSchema, buildIndex, insertChunk, getChunkCount } = require('./db.js');

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const PDF_PATH = process.env.PDF_PATH || process.argv[2];
const CHUNK_SIZE = 1400;
const CHUNK_OVERLAP = 200;

function detectCategory(text) {
  const t = text.toLowerCase();
  if (t.includes('whatsapp') || t.includes('advanced messaging') || t.includes('/messages/notifications')) return 'advanced-messaging';
  if (t.includes('/sms') || t.includes('sms message') || t.includes('smsmessage')) return 'sms';
  if (t.includes('/chat') && (t.includes('chatthread') || t.includes('chat thread') || t.includes('/messages'))) return 'chat';
  if (t.includes('call automation') || t.includes('callautomation') || t.includes('pstn') || t.includes('voip')) return 'call-automation';
  if (t.includes('communication identity') || t.includes('access token') || t.includes('identities') || t.includes('communicationidentity')) return 'identity';
  if (t.includes('/email') || t.includes('emailmessage') || t.includes('send email')) return 'email';
  if (t.includes('phone number') || t.includes('phonenumber') || t.includes('/availablephonenumbers')) return 'phone-numbers';
  if (t.includes('/rooms') || t.includes('roomparticipant') || t.includes('communication room')) return 'rooms';
  if (t.includes('network traversal') || t.includes('ice') || t.includes('relay')) return 'network-traversal';
  if (t.includes('authentication') || t.includes('hmac') || t.includes('access key')) return 'auth';
  return 'general';
}

function chunkText(text, chunkSizeChars = CHUNK_SIZE * 4, overlapChars = CHUNK_OVERLAP * 4) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    let end = start + chunkSizeChars;
    if (end < text.length) {
      const paraBreak = text.lastIndexOf('\n\n', end);
      if (paraBreak > start + chunkSizeChars * 0.5) {
        end = paraBreak + 2;
      } else {
        const lineBreak = text.lastIndexOf('\n', end);
        if (lineBreak > start + chunkSizeChars * 0.7) {
          end = lineBreak + 1;
        }
      }
    }
    const chunk = text.slice(start, end).trim();
    if (chunk.length > 100) chunks.push(chunk);
    start = end - overlapChars;
  }
  return chunks;
}

async function embedBatch(texts) {
  const resp = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: texts,
  });
  return resp.data.map(d => d.embedding);
}

async function main() {
  if (!PDF_PATH) {
    console.error('Usage: PDF_PATH=/path/to/file.pdf node ingest.js');
    process.exit(1);
  }
  if (!fs.existsSync(PDF_PATH)) {
    console.error(`PDF not found: ${PDF_PATH}`);
    process.exit(1);
  }

  console.log('Setting up database schema...');
  await setupSchema();

  const existing = await getChunkCount();
  if (existing > 0) {
    console.log(`Knowledge base already has ${existing} chunks. To re-ingest, truncate: psql $DATABASE_URL -c "TRUNCATE acs_chunks"`);
    process.exit(0);
  }

  console.log(`Extracting text from PDF: ${PDF_PATH}`);
  const txtPath = PDF_PATH.replace('.pdf', '.txt');
  if (!fs.existsSync(txtPath)) {
    execSync(`pdftotext -layout "${PDF_PATH}" "${txtPath}"`, { stdio: 'inherit', timeout: 600000 });
  } else {
    console.log(`Using cached text file: ${txtPath}`);
  }

  const text = fs.readFileSync(txtPath, 'utf8');
  console.log(`Extracted ${(text.length / 1024 / 1024).toFixed(1)} MB of text`);

  const chunks = chunkText(text);
  console.log(`Created ${chunks.length} chunks`);

  const BATCH_SIZE = 100;
  let inserted = 0;

  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    const embeddings = await embedBatch(batch);

    for (let j = 0; j < batch.length; j++) {
      const category = detectCategory(batch[j]);
      await insertChunk(batch[j], category, null, null, embeddings[j]);
    }

    inserted += batch.length;
    const pct = ((inserted / chunks.length) * 100).toFixed(1);
    console.log(`Progress: ${inserted}/${chunks.length} chunks (${pct}%)`);
  }

  console.log(`Done! Inserted ${inserted} chunks.`);
  console.log('Building HNSW index...');
  await buildIndex();
  console.log('All done! ACS knowledge base ready.');
  process.exit(0);
}

main().catch(err => {
  console.error('Ingest failed:', err);
  process.exit(1);
});
