import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import express from 'express';
import multer from 'multer';
import axios from 'axios';
import FormData from 'form-data';
import cors from 'cors';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PROJECT_ROOT = path.resolve(__dirname, '..');
const PUBLIC_DIR = path.join(__dirname, 'public');
const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 300 * 1024 * 1024 } });

const PYTHON_SERVER_URL = process.env.PYTHON_SERVER_URL || 'http://127.0.0.1:8000';
app.use(cors());
app.use(express.json());
app.use(express.static(PUBLIC_DIR));

function resolveJobDir(outputDir) {
  const cleaned = outputDir.startsWith('./') ? outputDir.slice(2) : outputDir;
  return path.resolve(PROJECT_ROOT, cleaned);
}

function toAbsolute(outputDir, relativePath) {
  if (!relativePath) return null;
  const baseDir = resolveJobDir(outputDir);
  const absPath = path.resolve(baseDir, relativePath);
  if (!absPath.startsWith(baseDir)) {
    throw new Error('Invalid path traversal attempt');
  }
  return absPath;
}

async function fetchJob(jobId) {
  const response = await axios.get(`${PYTHON_SERVER_URL}/jobs/${jobId}`);
  return response.data;
}

app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }
    const form = new FormData();
    form.append('file', req.file.buffer, { filename: req.file.originalname, contentType: req.file.mimetype });
    const response = await axios.post(`${PYTHON_SERVER_URL}/jobs`, form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });
    res.json(response.data);
  } catch (err) {
    console.error('Upload error:', err.message);
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    res.status(500).json({ error: 'Failed to queue conversion job' });
  }
});

app.post('/api/import-json', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No JSON file uploaded' });
    }
    const form = new FormData();
    form.append('file', req.file.buffer, {
      filename: req.file.originalname || 'document.json',
      contentType: req.file.mimetype || 'application/json',
    });
    const response = await axios.post(`${PYTHON_SERVER_URL}/jobs/import-json`, form, {
      headers: form.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });
    res.json(response.data);
  } catch (err) {
    console.error('JSON import error:', err.message);
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    res.status(500).json({ error: 'Failed to import JSON' });
  }
});

app.get('/api/jobs/:id/status', async (req, res) => {
  try {
    const job = await fetchJob(req.params.id);
    res.json(job);
  } catch (err) {
    console.error('Status error:', err.message);
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    res.status(500).json({ error: 'Unable to fetch job status' });
  }
});

app.get('/api/jobs/:id/file', async (req, res) => {
  const { type, name } = req.query;
  if (!type) {
    return res.status(400).json({ error: 'Missing required query parameter: type' });
  }

  try {
    const job = await fetchJob(req.params.id);
    if (job.status !== 'completed') {
      return res.status(409).json({ error: 'Job is not complete' });
    }

    const outputDir = job.output_dir;
    let relativePath;

    switch (type) {
      case 'json':
        relativePath = job.result?.json_path;
        break;
      case 'html':
        relativePath = job.result?.html_path;
        break;
      case 'metadata':
        relativePath = job.result?.metadata_path;
        break;
      case 'asset':
        if (!name) {
          return res.status(400).json({ error: 'Missing asset name' });
        }
        relativePath = job.result?.assets?.find((asset) => asset === name);
        break;
      default:
        return res.status(400).json({ error: `Unsupported type ${type}` });
    }

    if (!relativePath) {
      return res.status(404).json({ error: 'Requested file not found' });
    }

    const absolutePath = toAbsolute(outputDir, relativePath);
    if (!fs.existsSync(absolutePath)) {
      return res.status(404).json({ error: 'File missing on server' });
    }

    res.sendFile(absolutePath);
  } catch (err) {
    console.error('File fetch error:', err.message);
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    res.status(500).json({ error: 'Unable to retrieve file' });
  }
});

app.get('/api/jobs/:id/assets', async (req, res) => {
  try {
    const job = await fetchJob(req.params.id);
    if (job.status !== 'completed') {
      return res.status(409).json({ error: 'Job is not complete' });
    }
    res.json({ assets: job.result?.assets ?? [] });
  } catch (err) {
    console.error('Asset list error:', err.message);
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    res.status(500).json({ error: 'Unable to list assets' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Marker reviewer web server running on http://localhost:${PORT}`);
  console.log(`Proxying GPU jobs to ${PYTHON_SERVER_URL}`);
});
