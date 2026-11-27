import React, { useCallback, useEffect, useState } from 'react';
import {
  EvaluationResponse,
  HealthResponse,
  HistoryEntry,
  ModelInfoResponse,
  PredictionResponse,
  RecognitionMode,
  SampleItem,
} from './types';
import { DrawingCanvas } from './components/DrawingCanvas';
import { PredictionPanel } from './components/PredictionPanel';
import { SampleGalleryAndHistory } from './components/SampleGalleryAndHistory';
import { ModelAnalyticsPanel } from './components/ModelAnalyticsPanel';

const HISTORY_STORAGE_KEY = 'digitai_prediction_history_v1';

export default function App() {
  const [activeTab, setActiveTab] = useState<'workspace' | 'analytics'>('workspace');
  const [mode, setMode] = useState<RecognitionMode>('multi');
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [backendConnected, setBackendConnected] = useState<boolean>(true);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [sourcePreviewUrl, setSourcePreviewUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);

  const [evalLoading, setEvalLoading] = useState<boolean>(false);
  const [evalResult, setEvalResult] = useState<EvaluationResponse | null>(null);

  const [history, setHistory] = useState<HistoryEntry[]>(() => {
    try {
      const saved = localStorage.getItem(HISTORY_STORAGE_KEY);
      return saved ? (JSON.parse(saved) as HistoryEntry[]) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch {
      // Ignore storage quota errors
    }
  }, [history]);

  const loadMetadata = useCallback(() => {
    fetch('/api/model-info')
      .then((res) => res.json())
      .then((data: ModelInfoResponse) => setModelInfo(data))
      .catch(() => {});

    fetch('/api/samples')
      .then((res) => res.json())
      .then((data: { samples: SampleItem[] }) => setSamples(data.samples ?? []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timerId: number | undefined;
    let wasDisconnected = false;

    const pollHealth = async () => {
      try {
        const res = await fetch('/api/health');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as HealthResponse;
        if (cancelled) return;
        setBackendConnected(true);
        setHealth(data);
        if (data.ready) {
          if (wasDisconnected || !modelInfo || samples.length === 0) {
            loadMetadata();
          }
          wasDisconnected = false;
          timerId = window.setTimeout(pollHealth, 15000);
        } else if (data.model_state !== 'error') {
          timerId = window.setTimeout(pollHealth, 2500);
        }
      } catch {
        if (!cancelled) {
          wasDisconnected = true;
          setBackendConnected(false);
          timerId = window.setTimeout(pollHealth, 3000);
        }
      }
    };

    pollHealth();
    return () => {
      cancelled = true;
      if (timerId !== undefined) window.clearTimeout(timerId);
    };
  }, [loadMetadata, modelInfo, samples.length]);

  const modelReady = backendConnected && Boolean(health?.ready || modelInfo?.status === 'loaded');

  const recordHistory = useCallback(
    (res: PredictionResponse, source: 'canvas' | 'upload' | 'sample') => {
      const entry: HistoryEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        predicted_number: res.predicted_number,
        mode: res.mode,
        confidence: res.confidence,
        digit_count: res.digit_count,
        processing_time_ms: res.processing_time_ms,
        source,
        processed_preview_base64: res.processed_preview_base64,
        low_confidence_warning: res.low_confidence_warning,
        segmentation_uncertain: res.segmentation_uncertain,
      };
      setHistory((prev) => [entry, ...prev].slice(0, 20));
    },
    []
  );

  const runBase64Prediction = useCallback(
    async (dataUrl: string, overrideMode?: RecognitionMode, source: 'canvas' | 'sample' = 'canvas') => {
      if (!modelReady) {
        setErrorMessage(
          !backendConnected
            ? 'Backend disconnected. Reconnecting automatically...'
            : health?.message || 'Initializing CNN model.'
        );
        return;
      }
      setIsPredicting(true);
      setErrorMessage(null);
      setSourcePreviewUrl(dataUrl);
      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: dataUrl, mode: overrideMode ?? mode }),
        });
        const data = await res.json();
        if (!res.ok) {
          setPrediction(null);
          setErrorMessage(data.error || 'Unable to process this image. Please try another image.');
          return;
        }
        setPrediction(data as PredictionResponse);
        recordHistory(data as PredictionResponse, source);
      } catch {
        setPrediction(null);
        setErrorMessage('Unable to process this image. Please try another image.');
      } finally {
        setIsPredicting(false);
      }
    },
    [backendConnected, health?.message, mode, modelReady, recordHistory]
  );

  const handleUploadFile = useCallback(
    async (file: File) => {
      if (!modelReady) {
        setErrorMessage(
          !backendConnected
            ? 'Backend disconnected. Reconnecting automatically...'
            : health?.message || 'Initializing CNN model.'
        );
        return;
      }
      setIsPredicting(true);
      setErrorMessage(null);

      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === 'string') {
          setSourcePreviewUrl(reader.result);
        }
      };
      reader.readAsDataURL(file);

      try {
        const form = new FormData();
        form.append('image', file);
        form.append('mode', mode);
        const res = await fetch('/api/predict', {
          method: 'POST',
          body: form,
        });
        const data = await res.json();
        if (!res.ok) {
          setPrediction(null);
          setErrorMessage(data.error || 'Unable to process this image. Please try another image.');
          return;
        }
        setPrediction(data as PredictionResponse);
        recordHistory(data as PredictionResponse, 'upload');
      } catch {
        setPrediction(null);
        setErrorMessage('Unable to process this image. Please try another image.');
      } finally {
        setIsPredicting(false);
      }
    },
    [backendConnected, health?.message, mode, modelReady, recordHistory]
  );

  const handleSelectSample = useCallback(
    (sample: SampleItem) => {
      setMode(sample.mode);
      setActiveTab('workspace');
      runBase64Prediction(sample.image_base64, sample.mode, 'sample');
    },
    [runBase64Prediction]
  );

  const handleRunEvaluation = useCallback(async (numSamples: number) => {
    setEvalLoading(true);
    try {
      const res = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_samples: numSamples }),
      });
      const data = await res.json();
      if (res.ok) {
        setEvalResult(data as EvaluationResponse);
        return data as EvaluationResponse;
      }
    } catch {
      // Ignore transient error
    } finally {
      setEvalLoading(false);
    }
    return null;
  }, []);

  const enh = modelInfo?.metrics?.enhanced_cnn;

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30">
        <div className="max-w-[1360px] mx-auto px-6 h-15 flex items-center justify-between gap-8">
          <a
            href="#top"
            onClick={(e) => {
              e.preventDefault();
              setActiveTab('workspace');
            }}
            className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
          >
            DigitAI
          </a>

          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600">
            <button
              type="button"
              onClick={() => setActiveTab('workspace')}
              className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === 'workspace' ? 'text-slate-900 underline underline-offset-8 decoration-sky-600 decoration-2' : ''
              }`}
            >
              Workspace
            </button>
            <a
              href="#samples"
              onClick={() => setActiveTab('workspace')}
              className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0"
            >
              Samples
            </a>
            <button
              type="button"
              onClick={() => setActiveTab('analytics')}
              className={`hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === 'analytics' ? 'text-slate-900 underline underline-offset-8 decoration-sky-600 decoration-2' : ''
              }`}
            >
              Model Metrics
            </button>
            <button
              type="button"
              onClick={() => setActiveTab('analytics')}
              className="hover:text-slate-900 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              CNN Architecture
            </button>
          </nav>

          <div className="flex items-center shrink-0">
            <button
              type="button"
              disabled={!modelReady}
              onClick={() => {
                setActiveTab('analytics');
                handleRunEvaluation(1000);
              }}
              className="px-4 py-2 text-xs font-semibold text-white bg-slate-900 rounded-lg hover:bg-slate-800 disabled:opacity-45 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              Run MNIST Test
            </button>
          </div>
        </div>
      </header>

      <main id="top" className="flex-1 max-w-[1360px] w-full mx-auto px-6 py-6 flex flex-col gap-6">
        <section className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-5">
          <div className="max-w-2xl">
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-slate-900 text-balance">
              Handwritten Digit Recognition Using Convolutional Neural Networks
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 mt-1">
              Single, double, and multi-digit segmentation pipeline powered by MNIST-trained CNN classification.
            </p>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-5 border-t lg:border-t-0 lg:border-l border-slate-200 pt-4 lg:pt-0 lg:pl-6 font-mono tabular-nums shrink-0">
            <div>
              <div className="font-sans text-[11px] text-slate-500">MNIST Test Accuracy</div>
              <div className="text-lg font-bold text-slate-900">
                {enh ? `${(enh.test_accuracy * 100).toFixed(2)}%` : '—'}
              </div>
            </div>
            <div>
              <div className="font-sans text-[11px] text-slate-500">Macro F1-Score</div>
              <div className="text-lg font-bold text-slate-900">
                {enh ? `${(enh.f1_macro * 100).toFixed(2)}%` : '—'}
              </div>
            </div>
            <div>
              <div className="font-sans text-[11px] text-slate-500">Model Parameters</div>
              <div className="text-lg font-bold text-slate-900">
                {modelInfo?.metrics?.parameters ? modelInfo.metrics.parameters.toLocaleString() : '—'}
              </div>
            </div>
            <div>
              <div className="font-sans text-[11px] text-slate-500">Model &amp; Backend</div>
              <div
                className={`text-sm font-semibold mt-0.5 ${
                  !backendConnected
                    ? 'text-amber-700'
                    : modelReady
                    ? 'text-emerald-700'
                    : health?.model_state === 'error'
                    ? 'text-rose-700'
                    : 'text-amber-700'
                }`}
              >
                {!backendConnected
                  ? 'Reconnecting...'
                  : modelReady
                  ? 'CNN model ready.'
                  : health?.model_state === 'training'
                  ? 'Training CNN...'
                  : health?.model_state === 'error'
                  ? 'Init Failed'
                  : 'Initializing...'}
              </div>
            </div>
          </div>
        </section>

        {(!backendConnected || !modelReady) && (
          <div
            role="status"
            className={`rounded-xl border px-4 py-3 text-xs sm:text-sm font-medium ${
              !backendConnected || health?.model_state !== 'error'
                ? 'bg-amber-50 border-amber-200 text-amber-900'
                : 'bg-rose-50 border-rose-200 text-rose-900'
            }`}
          >
            {!backendConnected
              ? 'Backend connection lost. Attempting automatic reconnection...'
              : health?.message || 'Initializing CNN model.'}
          </div>
        )}

        <div className="flex md:hidden items-center gap-2">
          <button
            type="button"
            onClick={() => setActiveTab('workspace')}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg border ${
              activeTab === 'workspace'
                ? 'bg-slate-900 text-white border-slate-900'
                : 'bg-white text-slate-700 border-slate-200'
            }`}
          >
            Recognition Workspace
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('analytics')}
            className={`flex-1 py-2 text-xs font-semibold rounded-lg border ${
              activeTab === 'analytics'
                ? 'bg-slate-900 text-white border-slate-900'
                : 'bg-white text-slate-700 border-slate-200'
            }`}
          >
            Model &amp; Metrics
          </button>
        </div>

        {activeTab === 'workspace' ? (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
              <div className="lg:col-span-5">
                <DrawingCanvas
                  mode={mode}
                  onModeChange={setMode}
                  onPredictBase64={(dataUrl) => runBase64Prediction(dataUrl, mode, 'canvas')}
                  onUploadFile={handleUploadFile}
                  isLoading={isPredicting}
                  modelReady={modelReady}
                />
              </div>
              <div className="lg:col-span-7">
                <PredictionPanel
                  prediction={prediction}
                  sourcePreviewUrl={sourcePreviewUrl}
                  errorMessage={errorMessage}
                  isLoading={isPredicting}
                />
              </div>
            </div>

            <SampleGalleryAndHistory
              samples={samples}
              history={history}
              onSelectSample={handleSelectSample}
              onClearHistory={() => setHistory([])}
              isLoading={isPredicting || !modelReady}
            />
          </>
        ) : (
          <ModelAnalyticsPanel
            modelInfo={modelInfo}
            onRunEvaluation={handleRunEvaluation}
            evalLoading={evalLoading}
            evalResult={evalResult}
          />
        )}
      </main>

      <footer className="border-t border-slate-200 bg-white mt-8">
        <div className="max-w-[1360px] mx-auto px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-500">
          <span>DigitAI · Handwritten Digit Recognition Using CNN</span>
          <span>MNIST 28×28 Classifier · OpenCV Contour &amp; Projection Segmentation</span>
        </div>
      </footer>
    </div>
  );
}
