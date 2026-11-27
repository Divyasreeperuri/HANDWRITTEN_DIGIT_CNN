import React, { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, Info, XCircle } from 'lucide-react';
import { PredictionResponse } from '../types';
import { ProbabilityChart } from './ProbabilityChart';

interface PredictionPanelProps {
  prediction: PredictionResponse | null;
  sourcePreviewUrl: string | null;
  errorMessage: string | null;
  isLoading: boolean;
}

export const PredictionPanel: React.FC<PredictionPanelProps> = ({
  prediction,
  sourcePreviewUrl,
  errorMessage,
  isLoading,
}) => {
  const [selectedDigitIdx, setSelectedDigitIdx] = useState<number>(0);

  useEffect(() => {
    setSelectedDigitIdx(0);
  }, [prediction]);

  const activeDigit = prediction?.digits?.[selectedDigitIdx] ?? prediction?.digits?.[0];

  return (
    <section className="bg-white border border-slate-200 rounded-xl p-5 flex flex-col gap-5">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Recognition Output</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Segmented left-to-right digit classification, normalized 28×28 inputs, and softmax scores.
          </p>
        </div>
        {prediction && (
          <div className="text-xs font-mono tabular-nums text-slate-500">
            {prediction.processing_time_ms.toFixed(1)} ms · {prediction.digit_count}{' '}
            {prediction.digit_count === 1 ? 'digit' : 'digits'}
          </div>
        )}
      </div>

      {errorMessage && (
        <div
          role="alert"
          className="flex items-start gap-2.5 p-3.5 rounded-lg border border-rose-200 bg-rose-50 text-rose-900 text-xs"
        >
          <XCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">Unable to recognize digits</div>
            <div className="text-rose-800 mt-0.5">{errorMessage}</div>
          </div>
        </div>
      )}

      {prediction && (prediction.low_confidence_warning || prediction.segmentation_uncertain || prediction.warnings.length > 0) && (
        <div
          role="status"
          className="flex items-start gap-2.5 p-3.5 rounded-lg border border-amber-200 bg-amber-50 text-amber-950 text-xs"
        >
          <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-semibold">
              {prediction.segmentation_uncertain && prediction.low_confidence_warning
                ? 'Uncertain segmentation & low prediction confidence'
                : prediction.segmentation_uncertain
                ? 'Segmentation uncertainty detected'
                : 'Low prediction confidence'}
            </div>
            {prediction.warnings.map((w, i) => (
              <div key={i} className="text-amber-900">
                {w}
              </div>
            ))}
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="py-16 flex flex-col items-center justify-center text-slate-500 gap-2 border border-slate-200 rounded-lg bg-slate-50">
          <div className="w-6 h-6 border-2 border-slate-300 border-t-sky-600 rounded-full animate-spin" />
          <span className="text-xs font-medium">Preprocessing, segmenting, and running CNN inference...</span>
        </div>
      ) : !prediction ? (
        <div className="py-14 px-6 flex flex-col items-center justify-center text-center border border-slate-200 rounded-lg bg-slate-50/70">
          <Info className="w-5 h-5 text-slate-400 mb-2" />
          <div className="text-sm font-medium text-slate-700">Ready for recognition</div>
          <p className="text-xs text-slate-500 mt-1 max-w-md">
            Draw a single digit, a two-digit pair, or a multi-digit sequence on the left canvas, upload a photo, or click a sample from the gallery below.
          </p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch">
            <div className="md:col-span-4 border border-slate-200 rounded-lg bg-slate-50 p-4 flex flex-col justify-between">
              <div className="text-xs font-medium text-slate-500">Predicted Number</div>
              <div className="my-3 text-5xl font-bold font-mono tabular-nums tracking-tight text-slate-900 break-all">
                {prediction.predicted_number}
              </div>
              <div className="space-y-1 text-xs text-slate-600 font-mono tabular-nums border-t border-slate-200 pt-3">
                <div className="flex items-center justify-between">
                  <span className="font-sans text-slate-500">Overall confidence</span>
                  <span className="font-semibold text-slate-900">
                    {(prediction.confidence * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="font-sans text-slate-500">Mode</span>
                  <span className="text-slate-800">{prediction.mode}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="font-sans text-slate-500">Color polarity</span>
                  <span className="text-slate-800">
                    {prediction.inverted_colors ? 'Inverted (Light BG)' : 'Dark BG'}
                  </span>
                </div>
              </div>
            </div>

            <div className="md:col-span-8 flex flex-col justify-between gap-3">
              <ProbabilityChart
                probabilities={activeDigit?.probabilities ?? prediction.primary_probabilities}
                predictedDigit={activeDigit?.digit}
                title={
                  prediction.digit_count > 1 && activeDigit
                    ? `Softmax Probabilities — Digit #${activeDigit.index + 1} (Predicted: ${activeDigit.digit})`
                    : 'Softmax Probability Distribution (Digits 0–9)'
                }
              />
            </div>
          </div>

          <div className="border-t border-slate-200 pt-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold text-slate-800">
                Individual Segmented Digits ({prediction.digits.length})
              </h3>
              <span className="text-xs text-slate-500">Click any digit tile to inspect its 0–9 probabilities</span>
            </div>

            <div className="flex flex-wrap gap-3">
              {prediction.digits.map((d, idx) => {
                const isSelected = idx === selectedDigitIdx;
                const isLow = d.confidence < 0.72;
                return (
                  <button
                    key={d.index}
                    type="button"
                    onClick={() => setSelectedDigitIdx(idx)}
                    className={`flex items-center gap-3 p-2.5 rounded-lg border text-left transition-colors cursor-pointer ${
                      isSelected
                        ? 'border-sky-500 bg-sky-50/50'
                        : 'border-slate-200 bg-white hover:border-slate-300'
                    }`}
                  >
                    <img
                      src={d.processed_digit_base64}
                      alt={`Normalized 28x28 digit ${d.digit}`}
                      className="w-11 h-11 rounded border border-slate-300 bg-black pixelated shrink-0"
                    />
                    <div className="min-w-[82px]">
                      <div className="flex items-baseline gap-1.5">
                        <span className="text-lg font-bold font-mono tabular-nums text-slate-900">
                          {d.digit}
                        </span>
                        <span className="text-xs text-slate-400 font-mono">#{d.index + 1}</span>
                      </div>
                      <div
                        className={`text-xs font-mono tabular-nums flex items-center gap-1 ${
                          isLow ? 'text-amber-700 font-medium' : 'text-emerald-700'
                        }`}
                      >
                        {!isLow && <CheckCircle2 className="w-3 h-3 shrink-0" />}
                        {isLow && <AlertTriangle className="w-3 h-3 shrink-0" />}
                        {(d.confidence * 100).toFixed(1)}%
                      </div>
                      <div className="text-[11px] text-slate-400 font-mono tabular-nums">
                        {d.bbox.w}×{d.bbox.h}px{d.was_split ? ' · split' : ''}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="border-t border-slate-200 pt-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <div className="text-xs font-medium text-slate-700 mb-2">Source Input Preview</div>
              <div className="border border-slate-200 rounded-lg bg-slate-50 h-40 flex items-center justify-center overflow-hidden p-2">
                {sourcePreviewUrl ? (
                  <img
                    src={sourcePreviewUrl}
                    alt="Original handwriting input"
                    className="max-h-full max-w-full object-contain"
                  />
                ) : (
                  <span className="text-xs text-slate-400">No source preview</span>
                )}
              </div>
            </div>
            <div>
              <div className="text-xs font-medium text-slate-700 mb-2">
                Processed &amp; Segmented Bounding Boxes
              </div>
              <div className="border border-slate-200 rounded-lg bg-slate-950 h-40 flex items-center justify-center overflow-hidden p-2">
                {prediction.processed_preview_base64 ? (
                  <img
                    src={prediction.processed_preview_base64}
                    alt="Processed binary image with digit bounding boxes"
                    className="max-h-full max-w-full object-contain"
                  />
                ) : (
                  <span className="text-xs text-slate-500">No processed preview</span>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
};
