import React, { useEffect, useRef, useState } from 'react';
import { Eraser, Play, RotateCcw, Upload } from 'lucide-react';
import { RecognitionMode } from '../types';

interface DrawingCanvasProps {
  mode: RecognitionMode;
  onModeChange: (mode: RecognitionMode) => void;
  onPredictBase64: (dataUrl: string) => void;
  onUploadFile: (file: File) => void;
  isLoading: boolean;
  modelReady: boolean;
}

export const DrawingCanvas: React.FC<DrawingCanvasProps> = ({
  mode,
  onModeChange,
  onPredictBase64,
  onUploadFile,
  isLoading,
  modelReady,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const historyRef = useRef<ImageData[]>([]);
  const isDrawingRef = useRef(false);

  const [brushSize, setBrushSize] = useState<number>(15);
  const [hasInk, setHasInk] = useState<boolean>(false);
  const [canUndo, setCanUndo] = useState<boolean>(false);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [inkTheme, setInkTheme] = useState<'paper' | 'dark'>('paper');

  const initCanvas = (theme: 'paper' | 'dark' = inkTheme) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    ctx.fillStyle = theme === 'paper' ? '#ffffff' : '#0f172a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const initial = ctx.getImageData(0, 0, canvas.width, canvas.height);
    historyRef.current = [initial];
    setHasInk(false);
    setCanUndo(false);
  };

  useEffect(() => {
    initCanvas(inkTheme);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [inkTheme]);

  const getPointerCoords = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    canvas.setPointerCapture(e.pointerId);
    isDrawingRef.current = true;
    const { x, y } = getPointerCoords(e);

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 0.01, y + 0.01);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = inkTheme === 'paper' ? '#0f172a' : '#f8fafc';
    ctx.lineWidth = brushSize;
    ctx.stroke();
    setHasInk(true);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    const { x, y } = getPointerCoords(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const handlePointerUp = () => {
    if (!isDrawingRef.current) return;
    isDrawingRef.current = false;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    historyRef.current.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    if (historyRef.current.length > 30) {
      historyRef.current.shift();
    }
    setCanUndo(historyRef.current.length > 1);
  };

  const handleUndo = () => {
    const canvas = canvasRef.current;
    if (!canvas || historyRef.current.length <= 1) return;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return;

    historyRef.current.pop();
    const prev = historyRef.current[historyRef.current.length - 1];
    ctx.putImageData(prev, 0, 0);
    setCanUndo(historyRef.current.length > 1);
    setHasInk(historyRef.current.length > 1);
  };

  const handlePredict = () => {
    if (!modelReady || isLoading) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    onPredictBase64(canvas.toDataURL('image/png'));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!modelReady || isLoading) return;
    const file = e.target.files?.[0];
    if (file) {
      onUploadFile(file);
      e.target.value = '';
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (!modelReady || isLoading) return;
    const file = e.dataTransfer.files?.[0];
    if (file) {
      onUploadFile(file);
    }
  };

  return (
    <section className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col gap-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Handwriting Input Pad</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Draw with mouse, touch, or stylus — or drop a scanned digit image below.
          </p>
        </div>

        <div
          className="inline-flex items-center p-1 bg-slate-100 rounded-lg border border-slate-200"
          role="group"
          aria-label="Recognition mode"
        >
          {(
            [
              { id: 'single', label: 'Single (0–9)' },
              { id: 'double', label: 'Double' },
              { id: 'multi', label: 'Multi-Digit' },
            ] as const
          ).map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onModeChange(item.id)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                mode === item.id
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative rounded-lg overflow-hidden border border-slate-200 bg-slate-50">
        <canvas
          ref={canvasRef}
          width={640}
          height={280}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          className="w-full h-[250px] sm:h-[280px] block touch-none cursor-crosshair select-none"
        />
        {!hasInk && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <span
              className={`text-sm font-medium ${
                inkTheme === 'paper' ? 'text-slate-400' : 'text-slate-500'
              }`}
            >
              {mode === 'single'
                ? 'Draw a single digit (e.g. 7)'
                : mode === 'double'
                ? 'Draw two digits side by side (e.g. 48)'
                : 'Draw a multi-digit number (e.g. 2026)'}
            </span>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex items-center gap-2.5 text-xs text-slate-600">
            <span>Brush</span>
            <input
              type="range"
              min={6}
              max={28}
              value={brushSize}
              onChange={(e) => setBrushSize(Number(e.target.value))}
              className="w-24 accent-sky-600 cursor-pointer"
              aria-label="Brush size"
            />
            <span className="font-mono tabular-nums text-slate-800 w-9">{brushSize}px</span>
          </label>

          <div className="flex items-center gap-1 p-0.5 bg-slate-100 rounded-md border border-slate-200 text-xs">
            <button
              type="button"
              onClick={() => setInkTheme('paper')}
              className={`px-2 py-1 rounded transition-colors cursor-pointer whitespace-nowrap ${
                inkTheme === 'paper' ? 'bg-white text-slate-900 font-medium shadow-xs' : 'text-slate-600'
              }`}
            >
              Light Paper
            </button>
            <button
              type="button"
              onClick={() => setInkTheme('dark')}
              className={`px-2 py-1 rounded transition-colors cursor-pointer whitespace-nowrap ${
                inkTheme === 'dark' ? 'bg-slate-900 text-white font-medium shadow-xs' : 'text-slate-600'
              }`}
            >
              Dark Board
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleUndo}
            disabled={!canUndo || isLoading}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-40 cursor-pointer whitespace-nowrap shrink-0"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Undo
          </button>
          <button
            type="button"
            onClick={() => initCanvas(inkTheme)}
            disabled={isLoading}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 disabled:opacity-40 cursor-pointer whitespace-nowrap shrink-0"
          >
            <Eraser className="w-3.5 h-3.5" />
            Clear
          </button>
          <button
            type="button"
            onClick={handlePredict}
            disabled={isLoading || !modelReady}
            className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-sky-600 rounded-lg hover:bg-sky-700 disabled:opacity-50 transition-colors cursor-pointer whitespace-nowrap shrink-0"
          >
            <Play className="w-3.5 h-3.5" />
            {!modelReady ? 'Model Loading...' : isLoading ? 'Recognizing...' : 'Predict Digits'}
          </button>
        </div>
      </div>

      <div className="border-t border-slate-200 pt-4">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            if (modelReady && !isLoading) setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => {
            if (modelReady && !isLoading) fileInputRef.current?.click();
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if ((e.key === 'Enter' || e.key === ' ') && modelReady && !isLoading) {
              fileInputRef.current?.click();
            }
          }}
          className={`border border-dashed rounded-lg px-4 py-4 flex flex-col sm:flex-row items-center justify-between gap-3 transition-colors ${
            !modelReady || isLoading
              ? 'border-slate-200 bg-slate-100 opacity-60 cursor-not-allowed'
              : isDragOver
              ? 'border-sky-500 bg-sky-50/70 cursor-pointer'
              : 'border-slate-300 bg-slate-50/60 hover:border-slate-400 hover:bg-slate-50 cursor-pointer'
          }`}
        >
          <div className="flex items-center gap-3 text-left">
            <div className="p-2 rounded-lg bg-white border border-slate-200 text-slate-600 shrink-0">
              <Upload className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs font-medium text-slate-800">
                Drop handwritten image here or browse files
              </div>
              <div className="text-xs text-slate-500">
                Supports PNG, JPG, JPEG · Black ink on white paper or white ink on dark background
              </div>
            </div>
          </div>
          <span className="px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-md whitespace-nowrap shrink-0">
            Select Image
          </span>
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,.jpg,.jpeg,.webp,.bmp"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>
      </div>
    </section>
  );
};
