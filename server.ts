import express from 'express';
import fs from 'fs';
import http from 'http';
import path from 'path';
import multer from 'multer';
import { createServer as createViteServer } from 'vite';
import { NodeCnnEngine, ValidationError } from './src/server/cnnEngine.ts';

const ROOT_DIR = path.resolve(process.cwd());

function resolveServerPort(): number {
  const portFlagIdx = process.argv.indexOf('--port');
  if (portFlagIdx !== -1 && process.argv[portFlagIdx + 1]) {
    const parsed = Number(process.argv[portFlagIdx + 1]);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  if (process.env.APP_PORT) {
    const parsed = Number(process.env.APP_PORT);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  if (process.env.DEFAULT_APP_PORT) {
    const parsed = Number(process.env.DEFAULT_APP_PORT);
    if (Number.isFinite(parsed) && parsed > 0) return parsed;
  }
  return 3000;
}

const PORT = resolveServerPort();
const FLASK_URL = process.env.FLASK_BACKEND_URL || 'http://127.0.0.1:5001';

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
});

const ALLOWED_EXTS = new Set(['png', 'jpg', 'jpeg', 'webp', 'bmp']);

async function tryProxyFlaskJson(endpoint: string, init?: RequestInit): Promise<Response | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    const res = await fetch(`${FLASK_URL}${endpoint}`, {
      ...init,
      signal: controller.signal,
    });
    clearTimeout(timer);
    return res;
  } catch {
    return null;
  }
}

async function startServer() {
  const app = express();
  const httpServer = http.createServer(app);
  const engine = new NodeCnnEngine(ROOT_DIR);

  if (!engine.isLoaded()) {
    console.warn('[DigitAI] CNN weights not present at Express startup; initiating automatic setup.');
    engine.startAutoTraining();
  } else {
    console.info('[DigitAI] CNN weights loaded cleanly on Express startup.');
  }

  app.use(express.json({ limit: '15mb' }));
  app.use('/static', express.static(path.join(ROOT_DIR, 'static')));

  app.get('/api/health', async (_req, res) => {
    const flaskRes = await tryProxyFlaskJson('/api/health');
    if (flaskRes && flaskRes.ok) {
      const data = await flaskRes.json();
      return res.json({ ...data, runtime: 'flask+keras' });
    }
    return res.json({
      ...engine.getHealth(),
      runtime: 'node-cnn-engine',
    });
  });

  app.get('/api/model-info', async (_req, res) => {
    const flaskRes = await tryProxyFlaskJson('/api/model-info');
    if (flaskRes && flaskRes.ok) {
      return res.json(await flaskRes.json());
    }
    const health = engine.getHealth();
    const metrics = engine.getMetrics();
    return res.json({
      status: health.ready ? 'loaded' : health.model_state,
      ready: health.ready,
      model_state: health.model_state,
      message: health.message,
      model_format: health.model_format,
      model_path: 'models/digitai_cnn.keras',
      metrics,
      plots: {
        training_curves: '/static/plots/training_curves.png',
        confusion_matrix: '/static/plots/confusion_matrix.png',
      },
      sequence_note:
        'DigitAI uses an OpenCV contour + vertical-projection segmentation pipeline paired with a 28x28 CNN classifier. While accurate on separated and lightly touching digits, segmentation can struggle on heavily overlapping cursive numerals or noisy backgrounds; an end-to-end CRNN (CNN + BiLSTM + CTC loss) or Attention-based sequence model removes explicit segmentation.',
    });
  });

  app.get('/api/samples', async (_req, res) => {
    const flaskRes = await tryProxyFlaskJson('/api/samples');
    if (flaskRes && flaskRes.ok) {
      return res.json(await flaskRes.json());
    }
    return res.json({ samples: engine.getSamples() });
  });

  app.post('/api/predict', upload.single('image'), async (req, res) => {
    try {
      const mode = String(req.body?.mode || 'multi');

      if (req.file) {
        const ext = path.extname(req.file.originalname || '').replace('.', '').toLowerCase();
        if (ext && !ALLOWED_EXTS.has(ext)) {
          return res.status(400).json({
            error: 'Please upload a valid handwritten digit image.',
            code: 'INVALID_IMAGE',
          });
        }

        try {
          const form = new FormData();
          form.append('mode', mode);
          form.append('image', new Blob([new Uint8Array(req.file.buffer)]), req.file.originalname || 'upload.png');
          const flaskRes = await tryProxyFlaskJson('/api/predict', { method: 'POST', body: form });
          if (flaskRes) {
            const body = await flaskRes.json();
            return res.status(flaskRes.status).json(body);
          }
        } catch (proxyErr) {
          console.warn('[DigitAI] Flask proxy unavailable, using NodeCnnEngine:', proxyErr);
        }

        engine.ensureReady(true);
        const result = engine.predictBuffer(req.file.buffer, mode);
        return res.json(result);
      }

      const imagePayload = req.body?.image;
      if (!imagePayload || typeof imagePayload !== 'string') {
        return res.status(400).json({
          error: 'Please upload a valid handwritten digit image.',
          code: 'INVALID_IMAGE',
        });
      }

      const flaskRes = await tryProxyFlaskJson('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imagePayload, mode }),
      });
      if (flaskRes) {
        const body = await flaskRes.json();
        return res.status(flaskRes.status).json(body);
      }

      engine.ensureReady(true);
      const base64Data = imagePayload.includes(',') ? imagePayload.split(',', 2)[1] : imagePayload;
      const buf = Buffer.from(base64Data, 'base64');
      const result = engine.predictBuffer(buf, mode);
      return res.json(result);
    } catch (err) {
      if (err instanceof ValidationError) {
        return res.status(err.status).json({
          error: err.message,
          code: err.code,
          model_state: err.modelState,
        });
      }
      console.error('[DigitAI] Prediction error:', err);
      return res.status(500).json({
        error: 'Unable to process this image. Please try another image.',
        code: 'PREDICTION_ERROR',
      });
    }
  });

  app.post('/api/evaluate', async (req, res) => {
    const numSamples = Math.max(100, Math.min(10000, Number(req.body?.num_samples || 1000)));
    const flaskRes = await tryProxyFlaskJson('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_samples: numSamples }),
    });
    if (flaskRes && flaskRes.ok) {
      return res.json(await flaskRes.json());
    }

    try {
      engine.ensureReady(true);
    } catch (err) {
      if (err instanceof ValidationError) {
        return res.status(err.status).json({ error: err.message, code: err.code, model_state: err.modelState });
      }
    }

    const metrics = engine.getMetrics();
    if (!metrics) {
      return res.status(503).json({ error: 'Initializing CNN model.', code: 'MODEL_INITIALIZING' });
    }
    const enh = metrics.enhanced_cnn;
    return res.json({
      samples_evaluated: metrics.test_samples,
      accuracy: enh.test_accuracy,
      loss: enh.test_loss,
      precision_macro: enh.precision_macro,
      recall_macro: enh.recall_macro,
      f1_macro: enh.f1_macro,
      total_inference_ms: Number((enh.batch_inference_ms_per_digit * metrics.test_samples).toFixed(2)),
      ms_per_digit: enh.batch_inference_ms_per_digit,
      confusion_matrix: metrics.confusion_matrix,
    });
  });

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR === 'true' ? false : true,
        ws: { server: httpServer },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(ROOT_DIR, 'dist');
    if (fs.existsSync(distDir)) {
      app.use(express.static(distDir));
      app.get('*', (_req, res) => {
        res.sendFile(path.join(distDir, 'index.html'));
      });
    }
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`[DigitAI] Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
