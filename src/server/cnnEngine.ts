import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { PNG } from 'pngjs';
import jpeg from 'jpeg-js';

type ConvLayer = {
  type: 'conv2d';
  name: string;
  kernel_size: [number, number];
  padding: string;
  activation: string;
  weights: number[][][][];
  bias: number[];
};

type BatchNormLayer = {
  type: 'batch_norm';
  name: string;
  epsilon: number;
  gamma: number[];
  beta: number[];
  mean: number[];
  variance: number[];
};

type MaxPoolLayer = {
  type: 'maxpool2d';
  name: string;
  pool_size: [number, number];
};

type FlattenLayer = {
  type: 'flatten';
  name: string;
};

type DenseLayer = {
  type: 'dense';
  name: string;
  activation: string;
  weights: number[][];
  bias: number[];
};

type ExportedLayer = ConvLayer | BatchNormLayer | MaxPoolLayer | FlattenLayer | DenseLayer;

interface ExportedWeights {
  model_name: string;
  layers: ExportedLayer[];
}

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
  wasSplit?: boolean;
}

export type EngineState = 'loading' | 'training' | 'ready' | 'error';

export class ValidationError extends Error {
  status: number;
  code: string;
  modelState?: EngineState;
  constructor(message: string, status = 400, code = 'INVALID_IMAGE', modelState?: EngineState) {
    super(message);
    this.status = status;
    this.code = code;
    this.modelState = modelState;
  }
}

export class NodeCnnEngine {
  private weights: ExportedWeights | null = null;
  private metrics: Record<string, any> | null = null;
  private rootDir: string;
  private state: EngineState = 'loading';
  private statusMessage = 'Initializing CNN model.';
  private lastError: string | null = null;
  private trainingProcess: ReturnType<typeof spawn> | null = null;

  constructor(rootDir: string) {
    this.rootDir = path.resolve(rootDir);
    this.load();
  }

  private getWeightsPath(): string {
    return path.resolve(this.rootDir, 'models', 'cnn_weights.json');
  }

  private getMetricsPath(): string {
    return path.resolve(this.rootDir, 'models', 'model_metrics.json');
  }

  private hasSavedKerasOrWeights(): boolean {
    const candidates = [
      path.resolve(this.rootDir, 'models', 'digitai_cnn.keras'),
      path.resolve(this.rootDir, 'models', 'digitai_cnn.h5'),
      path.resolve(this.rootDir, 'models', 'digitai_cnn.weights.h5'),
    ];
    return candidates.some((p) => fs.existsSync(p) && fs.statSync(p).size > 0);
  }

  private resolvePythonExecutable(): string | null {
    const candidates = [
      path.resolve(this.rootDir, '.venv', 'bin', 'python'),
      path.resolve(this.rootDir, '.venv', 'Scripts', 'python.exe'),
      path.resolve(this.rootDir, 'venv', 'bin', 'python'),
      path.resolve(this.rootDir, 'venv', 'Scripts', 'python.exe'),
      '/tmp/pyvenv/bin/python',
      'python3',
      'python',
    ];
    for (const candidate of candidates) {
      if (candidate.includes(path.sep)) {
        if (fs.existsSync(candidate)) return candidate;
      } else {
        return candidate;
      }
    }
    return null;
  }

  public startAutoTraining(): boolean {
    if (this.trainingProcess) return true;
    const trainScript = path.resolve(this.rootDir, 'training', 'train_cnn.py');
    if (!fs.existsSync(trainScript)) {
      this.state = 'error';
      this.statusMessage = 'CNN model initialization failed.';
      this.lastError = `Missing training script at ${trainScript}`;
      console.error('[DigitAI]', this.lastError);
      return false;
    }

    const py = this.resolvePythonExecutable();
    if (!py) {
      this.state = 'error';
      this.statusMessage = 'CNN model initialization failed.';
      this.lastError = 'No Python executable available to initialize or train the CNN.';
      console.error('[DigitAI]', this.lastError);
      return false;
    }

    const canReconstructFromSaved = this.hasSavedKerasOrWeights();
    const args = canReconstructFromSaved
      ? [
          '-c',
          'import sys; from backend.model_service import ModelService; s = ModelService(auto_initialize=True, auto_train=True); assert s.is_ready()',
        ]
      : ['-u', trainScript];

    this.state = canReconstructFromSaved ? 'loading' : 'training';
    this.statusMessage = canReconstructFromSaved ? 'Initializing CNN model.' : 'Training CNN for the first time.';
    this.lastError = null;
    console.info(`[DigitAI] Starting Python CNN setup via ${py} (${canReconstructFromSaved ? 'reconstruct' : 'train'})`);

    const proc = spawn(py, args, {
      cwd: this.rootDir,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    this.trainingProcess = proc;

    let stderrBuf = '';
    proc.stdout?.on('data', (chunk) => {
      const text = chunk.toString().trim();
      if (text) console.info(`[DigitAI python] ${text}`);
    });
    proc.stderr?.on('data', (chunk) => {
      const text = chunk.toString().trim();
      if (text) {
        stderrBuf = (stderrBuf + '\n' + text).slice(-1200);
        console.warn(`[DigitAI python stderr] ${text}`);
      }
    });

    proc.on('error', (err) => {
      console.error('[DigitAI] Failed spawning Python CNN setup process:', err);
      this.trainingProcess = null;
      this.state = 'error';
      this.lastError = err.message;
      this.statusMessage = 'CNN model initialization failed.';
    });

    proc.on('exit', (code) => {
      this.trainingProcess = null;
      if (code === 0 && this.load()) {
        console.info('[DigitAI] CNN weights loaded and ready.');
      } else if (!this.isLoaded()) {
        this.state = 'error';
        this.lastError = `Python setup exited with code ${code}`;
        this.statusMessage = stderrBuf.includes('No module named')
          ? 'CNN model initialization failed. Missing Python dependencies (install requirements.txt).'
          : 'CNN model initialization failed.';
        console.error(`[DigitAI] ${this.lastError}`, stderrBuf);
      }
    });

    return true;
  }

  public load(): boolean {
    const weightsPath = this.getWeightsPath();
    const metricsPath = this.getMetricsPath();

    if (fs.existsSync(metricsPath)) {
      try {
        this.metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf-8')) as Record<string, any>;
      } catch (err) {
        console.error(`[DigitAI] Failed reading metrics from ${metricsPath}:`, err);
      }
    }

    if (fs.existsSync(weightsPath)) {
      try {
        const stat = fs.statSync(weightsPath);
        if (stat.size > 0) {
          const parsed = JSON.parse(fs.readFileSync(weightsPath, 'utf-8')) as ExportedWeights;
          if (!parsed || !Array.isArray(parsed.layers) || parsed.layers.length === 0) {
            throw new Error('Exported cnn_weights.json has no layers.');
          }
          this.weights = parsed;
          this.state = 'ready';
          this.statusMessage = 'CNN model ready.';
          this.lastError = null;
          return true;
        }
      } catch (err) {
        console.error(`[DigitAI] Failed loading CNN weights from ${weightsPath}:`, err);
        this.weights = null;
        this.lastError = err instanceof Error ? err.message : 'Invalid weights file';
      }
    }

    if (this.trainingProcess) {
      if (this.state !== 'training' && this.state !== 'loading') {
        this.state = 'training';
        this.statusMessage = 'Training CNN for the first time.';
      }
    } else if (!this.weights && this.state !== 'error') {
      this.state = 'loading';
      this.statusMessage = 'Initializing CNN model.';
    }
    return false;
  }

  public ensureReady(triggerTrainIfMissing = true): void {
    if (this.isLoaded()) return;
    if (this.load()) return;

    if (this.trainingProcess) {
      const msg = this.state === 'loading' ? 'Initializing CNN model.' : 'Training CNN for the first time.';
      throw new ValidationError(msg, 503, 'MODEL_TRAINING', this.state);
    }

    if (this.state === 'error') {
      throw new ValidationError(this.statusMessage || 'CNN model initialization failed.', 503, 'MODEL_UNAVAILABLE', 'error');
    }

    if (triggerTrainIfMissing) {
      const started = this.startAutoTraining();
      if (started) {
        throw new ValidationError(
          this.statusMessage || 'Training CNN for the first time.',
          503,
          'MODEL_TRAINING',
          this.state
        );
      }
    }

    throw new ValidationError(this.statusMessage || 'Initializing CNN model.', 503, 'MODEL_INITIALIZING', this.state);
  }

  public isLoaded(): boolean {
    return this.weights !== null && this.state === 'ready';
  }

  public getHealth(): Record<string, any> {
    if (!this.isLoaded()) {
      this.load();
    }
    const ready = this.isLoaded();
    return {
      status: ready ? 'ok' : this.state === 'training' ? 'training' : this.state === 'error' ? 'error' : 'initializing',
      ready,
      model_loaded: ready,
      model_state: this.state,
      message: this.statusMessage,
      model_name: this.metrics?.model_name || 'DigitAI Enhanced CNN',
      model_format: ready ? 'cnn_weights.json' : null,
      model_path: this.getWeightsPath(),
      test_accuracy: this.metrics?.enhanced_cnn?.test_accuracy ?? null,
    };
  }

  public getMetrics(): Record<string, any> | null {
    if (!this.metrics) {
      const metricsPath = this.getMetricsPath();
      if (fs.existsSync(metricsPath)) {
        try {
          this.metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf-8')) as Record<string, any>;
        } catch (err) {
          console.error('[DigitAI] Failed parsing metrics JSON:', err);
        }
      }
    }
    return this.metrics;
  }

  public getSamples(): Array<Record<string, any>> {
    const samplesDir = path.join(this.rootDir, 'static', 'samples');
    if (!fs.existsSync(samplesDir)) return [];

    const specs = [
      { id: 'sample_single_7', label: 'Single Digit: 7', expected: '7', mode: 'single', background: 'white' },
      { id: 'sample_single_3', label: 'Single Digit: 3', expected: '3', mode: 'single', background: 'dark' },
      { id: 'sample_double_48', label: 'Double Digit: 48', expected: '48', mode: 'double', background: 'white' },
      { id: 'sample_double_96', label: 'Double Digit: 96', expected: '96', mode: 'double', background: 'dark' },
      { id: 'sample_multi_2026', label: 'Multi Digit: 2026', expected: '2026', mode: 'multi', background: 'white' },
      { id: 'sample_multi_987654', label: 'Multi Digit: 987654', expected: '987654', mode: 'multi', background: 'white' },
    ];

    const out: Array<Record<string, any>> = [];
    for (const s of specs) {
      const filePath = path.join(samplesDir, `${s.id}.png`);
      if (fs.existsSync(filePath)) {
        const b64 = fs.readFileSync(filePath).toString('base64');
        out.push({
          ...s,
          image_url: `/static/samples/${s.id}.png`,
          image_base64: `data:image/png;base64,${b64}`,
        });
      }
    }
    return out;
  }

  private decodeImageBuffer(buf: Buffer): { gray: Uint8Array; width: number; height: number } {
    if (!buf || buf.length < 16) {
      throw new ValidationError('Please upload a valid handwritten digit image.', 400, 'INVALID_IMAGE');
    }

    let rgba: Uint8Array;
    let width: number;
    let height: number;

    if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4e && buf[3] === 0x47) {
      try {
        const parsed = PNG.sync.read(buf);
        rgba = parsed.data;
        width = parsed.width;
        height = parsed.height;
      } catch (err) {
        console.error('[DigitAI] PNG decode error:', err);
        throw new ValidationError('Please upload a valid handwritten digit image.', 400, 'INVALID_IMAGE');
      }
    } else if (buf[0] === 0xff && buf[1] === 0xd8) {
      try {
        const parsed = jpeg.decode(buf, { useTArray: true });
        rgba = parsed.data;
        width = parsed.width;
        height = parsed.height;
      } catch (err) {
        console.error('[DigitAI] JPEG decode error:', err);
        throw new ValidationError('Please upload a valid handwritten digit image.', 400, 'INVALID_IMAGE');
      }
    } else {
      throw new ValidationError('Please upload a valid handwritten digit image.', 400, 'INVALID_IMAGE');
    }

    if (width < 8 || height < 8) {
      throw new ValidationError('Please upload a valid handwritten digit image.', 400, 'INVALID_IMAGE');
    }

    let opaqueSum = 0;
    let opaqueCount = 0;
    let hasTransparency = false;
    for (let i = 0; i < width * height; i++) {
      const a = rgba[i * 4 + 3];
      if (a < 250) hasTransparency = true;
      if (a > 20) {
        const lum = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
        opaqueSum += lum;
        opaqueCount++;
      }
    }
    if (opaqueCount === 0) {
      throw new ValidationError('No handwritten digits detected.', 422, 'NO_DIGITS_DETECTED');
    }
    const bgVal = hasTransparency && opaqueSum / opaqueCount > 127 ? 0 : 255;

    const gray = new Uint8Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const r = rgba[i * 4];
      const g = rgba[i * 4 + 1];
      const b = rgba[i * 4 + 2];
      const a = rgba[i * 4 + 3] / 255;
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      const comp = hasTransparency ? lum * a + bgVal * (1 - a) : lum;
      gray[i] = Math.max(0, Math.min(255, Math.round(comp)));
    }

    return { gray, width, height };
  }

  private otsuThreshold(work: Uint8Array): number {
    const hist = new Int32Array(256);
    for (let i = 0; i < work.length; i++) hist[work[i]]++;
    const total = work.length;
    let sum = 0;
    for (let t = 0; t < 256; t++) sum += t * hist[t];

    let sumB = 0;
    let wB = 0;
    let maxVar = -1;
    let threshold = 127;

    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      const wF = total - wB;
      if (wF === 0) break;
      sumB += t * hist[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      const between = wB * wF * (mB - mF) * (mB - mF);
      if (between > maxVar) {
        maxVar = between;
        threshold = t;
      }
    }
    return threshold;
  }

  private encodeGrayToBase64Png(gray: Uint8Array, width: number, height: number): string {
    const png = new PNG({ width, height });
    for (let i = 0; i < width * height; i++) {
      const v = gray[i];
      png.data[i * 4] = v;
      png.data[i * 4 + 1] = v;
      png.data[i * 4 + 2] = v;
      png.data[i * 4 + 3] = 255;
    }
    const buf = PNG.sync.write(png);
    return `data:image/png;base64,${buf.toString('base64')}`;
  }

  private encodeAnnotatedPreview(binary: Uint8Array, width: number, height: number, boxes: BBox[]): string {
    const png = new PNG({ width, height });
    for (let i = 0; i < width * height; i++) {
      const v = binary[i];
      png.data[i * 4] = v;
      png.data[i * 4 + 1] = v;
      png.data[i * 4 + 2] = v;
      png.data[i * 4 + 3] = 255;
    }

    const drawRect = (bx: number, by: number, bw: number, bh: number, r: number, g: number, b: number) => {
      const x0 = Math.max(0, bx);
      const y0 = Math.max(0, by);
      const x1 = Math.min(width - 1, bx + bw - 1);
      const y1 = Math.min(height - 1, by + bh - 1);
      for (let t = 0; t < 2; t++) {
        for (let x = x0; x <= x1; x++) {
          for (const y of [Math.min(height - 1, y0 + t), Math.max(0, y1 - t)]) {
            const idx = (y * width + x) * 4;
            png.data[idx] = r;
            png.data[idx + 1] = g;
            png.data[idx + 2] = b;
          }
        }
        for (let y = y0; y <= y1; y++) {
          for (const x of [Math.min(width - 1, x0 + t), Math.max(0, x1 - t)]) {
            const idx = (y * width + x) * 4;
            png.data[idx] = r;
            png.data[idx + 1] = g;
            png.data[idx + 2] = b;
          }
        }
      }
    };

    for (const box of boxes) {
      if (box.wasSplit) {
        drawRect(box.x, box.y, box.w, box.h, 245, 158, 11);
      } else {
        drawRect(box.x, box.y, box.w, box.h, 6, 182, 212);
      }
    }

    const buf = PNG.sync.write(png);
    return `data:image/png;base64,${buf.toString('base64')}`;
  }

  private normalizeToMnist28x28(softWork: Uint8Array, binary: Uint8Array, imgW: number, box: BBox): Float32Array {
    let minX = box.x + box.w;
    let maxX = box.x - 1;
    let minY = box.y + box.h;
    let maxY = box.y - 1;

    for (let y = box.y; y < box.y + box.h; y++) {
      for (let x = box.x; x < box.x + box.w; x++) {
        if (binary[y * imgW + x] > 0) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (maxX < minX || maxY < minY) {
      return new Float32Array(28 * 28);
    }

    const rw = maxX - minX + 1;
    const rh = maxY - minY + 1;
    const scale = 20.0 / Math.max(rw, rh);
    const newW = Math.max(1, Math.round(rw * scale));
    const newH = Math.max(1, Math.round(rh * scale));

    const resized = new Float32Array(newW * newH);
    for (let ny = 0; ny < newH; ny++) {
      const sy = ((ny + 0.5) / newH) * rh - 0.5;
      const y0 = Math.max(0, Math.min(rh - 1, Math.floor(sy)));
      const y1 = Math.max(0, Math.min(rh - 1, y0 + 1));
      const fy = Math.max(0, Math.min(1, sy - y0));
      for (let nx = 0; nx < newW; nx++) {
        const sx = ((nx + 0.5) / newW) * rw - 0.5;
        const x0 = Math.max(0, Math.min(rw - 1, Math.floor(sx)));
        const x1 = Math.max(0, Math.min(rw - 1, x0 + 1));
        const fx = Math.max(0, Math.min(1, sx - x0));

        const p00 = softWork[(minY + y0) * imgW + (minX + x0)] / 255.0;
        const p10 = softWork[(minY + y0) * imgW + (minX + x1)] / 255.0;
        const p01 = softWork[(minY + y1) * imgW + (minX + x0)] / 255.0;
        const p11 = softWork[(minY + y1) * imgW + (minX + x1)] / 255.0;

        resized[ny * newW + nx] = (p00 * (1 - fx) + p10 * fx) * (1 - fy) + (p01 * (1 - fx) + p11 * fx) * fy;
      }
    }

    const canvas = new Float32Array(28 * 28);
    const padX = Math.floor((28 - newW) / 2);
    const padY = Math.floor((28 - newH) / 2);
    for (let y = 0; y < newH; y++) {
      for (let x = 0; x < newW; x++) {
        canvas[(padY + y) * 28 + (padX + x)] = resized[y * newW + x];
      }
    }

    let mass = 0;
    let sumX = 0;
    let sumY = 0;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = canvas[y * 28 + x];
        mass += v;
        sumX += x * v;
        sumY += y * v;
      }
    }

    const shifted = new Float32Array(28 * 28);
    if (mass > 1e-4) {
      const shiftX = Math.max(-4, Math.min(4, Math.round(13.5 - sumX / mass)));
      const shiftY = Math.max(-4, Math.min(4, Math.round(13.5 - sumY / mass)));
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          const sx = x - shiftX;
          const sy = y - shiftY;
          if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) {
            shifted[y * 28 + x] = canvas[sy * 28 + sx];
          }
        }
      }
    } else {
      shifted.set(canvas);
    }

    let peak = 0;
    for (let i = 0; i < shifted.length; i++) {
      if (shifted[i] > peak) peak = shifted[i];
    }
    if (peak > 0.05) {
      for (let i = 0; i < shifted.length; i++) {
        shifted[i] = Math.min(1.0, shifted[i] / peak);
      }
    }
    return shifted;
  }

  private forwardPass(input28x28: Float32Array): number[] {
    this.ensureReady(true);
    if (!this.weights) {
      throw new ValidationError('Initializing CNN model. Please wait.', 503, 'MODEL_INITIALIZING', this.state);
    }

    let h = 28;
    let w = 28;
    let c = 1;
    let tensor: Float32Array = input28x28;

    for (const layer of this.weights.layers) {
      if (layer.type === 'conv2d') {
        const kh = layer.kernel_size[0];
        const kw = layer.kernel_size[1];
        const outC = layer.bias.length;
        const padY = Math.floor(kh / 2);
        const padX = Math.floor(kw / 2);
        const out = new Float32Array(h * w * outC);

        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const outBase = (y * w + x) * outC;
            for (let oc = 0; oc < outC; oc++) {
              let acc = layer.bias[oc];
              for (let ky = 0; ky < kh; ky++) {
                const iy = y + ky - padY;
                if (iy < 0 || iy >= h) continue;
                const rowWeights = layer.weights[ky];
                for (let kx = 0; kx < kw; kx++) {
                  const ix = x + kx - padX;
                  if (ix < 0 || ix >= w) continue;
                  const inBase = (iy * w + ix) * c;
                  const kWeights = rowWeights[kx];
                  for (let ic = 0; ic < c; ic++) {
                    acc += tensor[inBase + ic] * kWeights[ic][oc];
                  }
                }
              }
              out[outBase + oc] = acc > 0 ? acc : 0;
            }
          }
        }
        tensor = out;
        c = outC;
      } else if (layer.type === 'batch_norm') {
        const scale = new Float32Array(c);
        const offset = new Float32Array(c);
        for (let ch = 0; ch < c; ch++) {
          scale[ch] = layer.gamma[ch] / Math.sqrt(layer.variance[ch] + layer.epsilon);
          offset[ch] = layer.beta[ch] - layer.mean[ch] * scale[ch];
        }
        for (let i = 0; i < h * w; i++) {
          const base = i * c;
          for (let ch = 0; ch < c; ch++) {
            tensor[base + ch] = tensor[base + ch] * scale[ch] + offset[ch];
          }
        }
      } else if (layer.type === 'maxpool2d') {
        const ph = layer.pool_size[0];
        const pw = layer.pool_size[1];
        const nh = Math.floor(h / ph);
        const nw = Math.floor(w / pw);
        const out = new Float32Array(nh * nw * c);
        for (let y = 0; y < nh; y++) {
          for (let x = 0; x < nw; x++) {
            const outBase = (y * nw + x) * c;
            for (let ch = 0; ch < c; ch++) {
              let maxV = -Infinity;
              for (let py = 0; py < ph; py++) {
                for (let px = 0; px < pw; px++) {
                  const v = tensor[((y * ph + py) * w + (x * pw + px)) * c + ch];
                  if (v > maxV) maxV = v;
                }
              }
              out[outBase + ch] = maxV;
            }
          }
        }
        tensor = out;
        h = nh;
        w = nw;
      } else if (layer.type === 'flatten') {
        // Keras NHWC order preserved
      } else if (layer.type === 'dense') {
        const inLen = tensor.length;
        const outLen = layer.bias.length;
        const out = new Float32Array(outLen);
        for (let j = 0; j < outLen; j++) {
          let acc = layer.bias[j];
          for (let i = 0; i < inLen; i++) {
            acc += tensor[i] * layer.weights[i][j];
          }
          out[j] = acc;
        }
        if (layer.activation === 'softmax') {
          let maxLogit = -Infinity;
          for (let j = 0; j < outLen; j++) {
            if (out[j] > maxLogit) maxLogit = out[j];
          }
          let sumExp = 0;
          const probs: number[] = [];
          for (let j = 0; j < outLen; j++) {
            const e = Math.exp(out[j] - maxLogit);
            probs.push(e);
            sumExp += e;
          }
          return probs.map((e) => e / sumExp);
        } else {
          for (let j = 0; j < outLen; j++) {
            out[j] = out[j] > 0 ? out[j] : 0;
          }
          tensor = out;
          h = 1;
          w = 1;
          c = outLen;
        }
      }
    }

    return Array.from(tensor);
  }

  private splitTouchingBox(
    binary: Uint8Array,
    width: number,
    box: BBox,
    forceDouble: boolean,
    depth = 0
  ): BBox[] {
    const aspect = box.w / Math.max(1, box.h);
    const shouldTry = aspect > 1.25 || (forceDouble && aspect > 0.98);
    if (!shouldTry || box.w < 18 || depth >= 3) {
      return [{ ...box, wasSplit: depth > 0 }];
    }

    const colSum = new Float32Array(box.w);
    for (let dx = 0; dx < box.w; dx++) {
      let s = 0;
      for (let dy = 0; dy < box.h; dy++) {
        if (binary[(box.y + dy) * width + (box.x + dx)] > 0) s++;
      }
      colSum[dx] = s;
    }

    const margin = Math.max(3, Math.floor(0.22 * box.w));
    if (box.w - 2 * margin < 3) {
      return [{ ...box, wasSplit: depth > 0 }];
    }

    let splitDx = margin;
    let minVal = Infinity;
    for (let dx = margin; dx < box.w - margin; dx++) {
      const smoothed =
        (colSum[Math.max(0, dx - 1)] + colSum[dx] + colSum[Math.min(box.w - 1, dx + 1)]) / 3.0;
      if (smoothed < minVal) {
        minVal = smoothed;
        splitDx = dx;
      }
    }

    let leftPeak = 1;
    let rightPeak = 1;
    for (let dx = 0; dx < splitDx; dx++) if (colSum[dx] > leftPeak) leftPeak = colSum[dx];
    for (let dx = splitDx + 1; dx < box.w; dx++) if (colSum[dx] > rightPeak) rightPeak = colSum[dx];
    const refPeak = Math.max(1, Math.min(leftPeak, rightPeak));

    if (forceDouble || minVal <= 0.48 * refPeak) {
      const ranges: Array<[number, number]> = [
        [box.x, box.x + splitDx],
        [box.x + splitDx, box.x + box.w],
      ];
      const parts: BBox[] = [];
      for (const [rx0, rx1] of ranges) {
        let minX = rx1;
        let maxX = rx0 - 1;
        let minY = box.y + box.h;
        let maxY = box.y - 1;
        let count = 0;
        for (let y = box.y; y < box.y + box.h; y++) {
          for (let x = rx0; x < rx1; x++) {
            if (binary[y * width + x] > 0) {
              count++;
              if (x < minX) minX = x;
              if (x > maxX) maxX = x;
              if (y < minY) minY = y;
              if (y > maxY) maxY = y;
            }
          }
        }
        if (count >= 6 && maxX - minX + 1 >= 3 && maxY - minY + 1 >= 6) {
          parts.push(
            ...this.splitTouchingBox(
              binary,
              width,
              { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 },
              false,
              depth + 1
            )
          );
        }
      }
      if (parts.length >= 2) return parts;
    }

    return [{ ...box, wasSplit: depth > 0 }];
  }

  public predictBuffer(buf: Buffer, modeInput = 'multi') {
    this.ensureReady(true);
    const t0 = performance.now();
    const mode = ['single', 'double', 'multi'].includes(modeInput) ? modeInput : 'multi';
    const { gray, width, height } = this.decodeImageBuffer(buf);

    let minVal = 255;
    let maxVal = 0;
    for (let i = 0; i < gray.length; i++) {
      if (gray[i] < minVal) minVal = gray[i];
      if (gray[i] > maxVal) maxVal = gray[i];
    }
    if (maxVal - minVal < 18) {
      throw new ValidationError('No handwritten digits detected.', 422, 'NO_DIGITS_DETECTED');
    }

    const border = Math.max(2, Math.floor(Math.min(width, height) / 20));
    const borderSamples: number[] = [];
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (y < border || y >= height - border || x < border || x >= width - border) {
          borderSamples.push(gray[y * width + x]);
        }
      }
    }
    borderSamples.sort((a, b) => a - b);
    const borderMedian = borderSamples[Math.floor(borderSamples.length / 2)] ?? 255;
    let inverted = borderMedian > 127;

    const work = new Uint8Array(gray.length);
    for (let i = 0; i < gray.length; i++) {
      work[i] = inverted ? 255 - gray[i] : gray[i];
    }

    const thresh = Math.max(24, this.otsuThreshold(work));
    const binary = new Uint8Array(gray.length);
    let fgCount = 0;
    for (let i = 0; i < work.length; i++) {
      if (work[i] > thresh) {
        binary[i] = 255;
        fgCount++;
      }
    }

    if (fgCount / gray.length > 0.65) {
      inverted = !inverted;
      fgCount = 0;
      for (let i = 0; i < binary.length; i++) {
        binary[i] = binary[i] === 255 ? 0 : 255;
        if (binary[i] === 255) fgCount++;
      }
    }

    if (fgCount < 8) {
      throw new ValidationError('No handwritten digits detected.', 422, 'NO_DIGITS_DETECTED');
    }

    const visited = new Uint8Array(gray.length);
    const minArea = Math.max(12, 0.00035 * width * height);
    const minH = Math.max(6, Math.floor(0.04 * height));
    const rawBoxes: BBox[] = [];
    const queue = new Int32Array(gray.length);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const startIdx = y * width + x;
        if (binary[startIdx] === 0 || visited[startIdx]) continue;
        let head = 0;
        let tail = 0;
        queue[tail++] = startIdx;
        visited[startIdx] = 1;

        let minX = x;
        let maxX = x;
        let minY = y;
        let maxY = y;
        let area = 0;

        while (head < tail) {
          const cur = queue[head++];
          const cy = Math.floor(cur / width);
          const cx = cur - cy * width;
          area++;
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;

          for (let dy = -1; dy <= 1; dy++) {
            const ny = cy + dy;
            if (ny < 0 || ny >= height) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const nx = cx + dx;
              if (nx < 0 || nx >= width) continue;
              const nIdx = ny * width + nx;
              if (binary[nIdx] > 0 && !visited[nIdx]) {
                visited[nIdx] = 1;
                queue[tail++] = nIdx;
              }
            }
          }
        }

        const bw = maxX - minX + 1;
        const bh = maxY - minY + 1;
        if (area >= minArea && (bh >= minH || bw >= minH)) {
          rawBoxes.push({ x: minX, y: minY, w: bw, h: bh });
        }
      }
    }

    if (rawBoxes.length === 0) {
      throw new ValidationError('No handwritten digits detected.', 422, 'NO_DIGITS_DETECTED');
    }

    let boxes = [...rawBoxes].sort((a, b) => a.x - b.x);
    let merged = true;
    while (merged) {
      merged = false;
      const next: BBox[] = [];
      const used = new Array(boxes.length).fill(false);
      for (let i = 0; i < boxes.length; i++) {
        if (used[i]) continue;
        let cur = { ...boxes[i] };
        for (let j = i + 1; j < boxes.length; j++) {
          if (used[j]) continue;
          const b = boxes[j];
          const overlapX = Math.max(0, Math.min(cur.x + cur.w, b.x + b.w) - Math.max(cur.x, b.x));
          const minW = Math.max(1, Math.min(cur.w, b.w));
          const vGap = Math.max(0, Math.max(cur.y, b.y) - Math.min(cur.y + cur.h, b.y + b.h));
          if (overlapX / minW >= 0.45 && vGap <= Math.max(10, Math.floor(0.2 * height))) {
            const nx = Math.min(cur.x, b.x);
            const ny = Math.min(cur.y, b.y);
            cur = {
              x: nx,
              y: ny,
              w: Math.max(cur.x + cur.w, b.x + b.w) - nx,
              h: Math.max(cur.y + cur.h, b.y + b.h) - ny,
            };
            used[j] = true;
            merged = true;
          }
        }
        used[i] = true;
        next.push(cur);
      }
      boxes = next.sort((a, b) => a.x - b.x);
    }

    const warnings: string[] = [];
    let segmentationUncertain = false;
    let finalBoxes: BBox[] = [];

    if (mode === 'single') {
      if (boxes.length > 1) {
        segmentationUncertain = true;
        warnings.push(
          `Detected ${boxes.length} separate regions in Single-Digit mode; combined primary region. Switch to Multi-Digit mode for multi-digit numbers.`
        );
        const nx = Math.min(...boxes.map((b) => b.x));
        const ny = Math.min(...boxes.map((b) => b.y));
        const mx = Math.max(...boxes.map((b) => b.x + b.w));
        const my = Math.max(...boxes.map((b) => b.y + b.h));
        finalBoxes = [{ x: nx, y: ny, w: mx - nx, h: my - ny, wasSplit: false }];
      } else {
        finalBoxes = [{ ...boxes[0], wasSplit: false }];
      }
    } else {
      const forceDouble = mode === 'double' && boxes.length === 1;
      for (const b of boxes) {
        finalBoxes.push(...this.splitTouchingBox(binary, width, b, forceDouble));
      }
      if (finalBoxes.some((b) => b.wasSplit)) {
        segmentationUncertain = true;
        warnings.push('Touching digits were detected and separated using vertical projection splitting.');
      }
    }

    finalBoxes.sort((a, b) => a.x - b.x);
    if (mode === 'double' && finalBoxes.length !== 2) {
      segmentationUncertain = true;
      warnings.push(`Double-Digit mode expected 2 digits, but segmentation found ${finalBoxes.length}.`);
    }

    const digitsOut = finalBoxes.map((box, idx) => {
      const mnist28 = this.normalizeToMnist28x28(work, binary, width, box);
      const probs = this.forwardPass(mnist28);
      let predDigit = 0;
      let maxP = -1;
      for (let d = 0; d < 10; d++) {
        if (probs[d] > maxP) {
          maxP = probs[d];
          predDigit = d;
        }
      }

      const cropGray = new Uint8Array(28 * 28);
      for (let i = 0; i < 28 * 28; i++) {
        cropGray[i] = Math.max(0, Math.min(255, Math.round(mnist28[i] * 255)));
      }

      const topAlternatives = probs
        .map((p, d) => ({ digit: d, probability: Number(p.toFixed(6)) }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3);

      return {
        index: idx,
        digit: predDigit,
        confidence: Number(maxP.toFixed(6)),
        probabilities: probs.map((p) => Number(p.toFixed(6))),
        top_alternatives: topAlternatives,
        bbox: { x: box.x, y: box.y, w: box.w, h: box.h },
        was_split: Boolean(box.wasSplit),
        processed_digit_base64: this.encodeGrayToBase64Png(cropGray, 28, 28),
      };
    });

    const lowConfidence = digitsOut.some((d) => d.confidence < 0.72);
    if (lowConfidence) {
      warnings.push('One or more digits had low prediction confidence (< 72%). Try writing darker, clearer strokes.');
    }

    const confProduct = digitsOut.reduce((acc, d) => acc * Math.max(1e-6, d.confidence), 1);
    const overallConf = digitsOut.length > 0 ? Math.pow(confProduct, 1 / digitsOut.length) : 0;
    const elapsedMs = performance.now() - t0;

    return {
      predicted_number: digitsOut.map((d) => String(d.digit)).join(''),
      mode,
      confidence: Number(overallConf.toFixed(6)),
      min_confidence: Number(Math.min(...digitsOut.map((d) => d.confidence)).toFixed(6)),
      digit_count: digitsOut.length,
      digits: digitsOut,
      primary_probabilities: digitsOut[0]?.probabilities ?? new Array(10).fill(0),
      low_confidence_warning: lowConfidence,
      segmentation_uncertain: segmentationUncertain,
      warnings,
      inverted_colors: inverted,
      processed_preview_base64: this.encodeAnnotatedPreview(binary, width, height, finalBoxes),
      binary_preview_base64: this.encodeGrayToBase64Png(binary, width, height),
      processing_time_ms: Number(elapsedMs.toFixed(2)),
    };
  }
}
