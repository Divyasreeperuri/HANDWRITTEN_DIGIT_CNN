import React, { useState } from 'react';
import { EvaluationResponse, ModelInfoResponse } from '../types';
import { Play } from 'lucide-react';

interface ModelAnalyticsPanelProps {
  modelInfo: ModelInfoResponse | null;
  onRunEvaluation: (numSamples: number) => Promise<EvaluationResponse | null>;
  evalLoading: boolean;
  evalResult: EvaluationResponse | null;
}

export const ModelAnalyticsPanel: React.FC<ModelAnalyticsPanelProps> = ({
  modelInfo,
  onRunEvaluation,
  evalLoading,
  evalResult,
}) => {
  const [evalCount, setEvalCount] = useState<number>(1000);

  if (!modelInfo?.metrics) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center text-sm text-slate-500">
        Loading trained CNN model metrics and architecture details...
      </div>
    );
  }

  const { enhanced_cnn: enh, baseline_cnn: base, confusion_matrix: activeCm, per_class_metrics, architecture_layers } =
    modelInfo.metrics;

  const displayCm = evalResult?.confusion_matrix ?? activeCm;
  const maxCmVal = Math.max(...displayCm.flat(), 1);

  const epochs = enh.history.accuracy.length;
  const chartW = 460;
  const chartH = 190;
  const padL = 40;
  const padR = 16;
  const padT = 16;
  const padB = 28;
  const innerW = chartW - padL - padR;
  const innerH = chartH - padT - padB;

  const toPolyline = (values: number[], minV: number, maxV: number) =>
    values
      .map((v, idx) => {
        const x = padL + (epochs <= 1 ? innerW / 2 : (idx / (epochs - 1)) * innerW);
        const norm = (v - minV) / Math.max(1e-6, maxV - minV);
        const y = padT + (1 - Math.max(0, Math.min(1, norm))) * innerH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  const accMin = Math.min(...enh.history.accuracy, ...enh.history.val_accuracy, 0.9);
  const lossMax = Math.max(...enh.history.loss, ...enh.history.val_loss, 0.25);

  return (
    <div className="flex flex-col gap-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Baseline vs. Enhanced CNN Evaluation
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Measured on the full {modelInfo.metrics.test_samples.toLocaleString()} MNIST test images.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-2 pr-3 font-medium">Metric</th>
                  <th className="py-2 px-3 font-medium text-right">Baseline CNN</th>
                  <th className="py-2 pl-3 font-medium text-right">Enhanced CNN</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                <tr>
                  <td className="py-2.5 pr-3 font-sans text-slate-700">Test Accuracy</td>
                  <td className="py-2.5 px-3 text-right text-slate-600">
                    {(base.test_accuracy * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 pl-3 text-right font-semibold text-sky-700">
                    {(enh.test_accuracy * 100).toFixed(2)}%
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-3 font-sans text-slate-700">Test Loss</td>
                  <td className="py-2.5 px-3 text-right text-slate-600">{base.test_loss.toFixed(4)}</td>
                  <td className="py-2.5 pl-3 text-right font-semibold text-slate-900">{enh.test_loss.toFixed(4)}</td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-3 font-sans text-slate-700">Macro Precision / Recall / F1</td>
                  <td className="py-2.5 px-3 text-right text-slate-600">
                    {(base.precision_macro * 100).toFixed(2)}% / {(base.recall_macro * 100).toFixed(2)}% /{' '}
                    {(base.f1_macro * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 pl-3 text-right font-semibold text-slate-900">
                    {(enh.precision_macro * 100).toFixed(2)}% / {(enh.recall_macro * 100).toFixed(2)}% /{' '}
                    {(enh.f1_macro * 100).toFixed(2)}%
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-3 font-sans text-slate-700">Trainable Parameters</td>
                  <td className="py-2.5 px-3 text-right text-slate-600">
                    {base.parameters.toLocaleString()}
                  </td>
                  <td className="py-2.5 pl-3 text-right text-slate-900">
                    {modelInfo.metrics.parameters.toLocaleString()}
                  </td>
                </tr>
                <tr>
                  <td className="py-2.5 pr-3 font-sans text-slate-700">Inference Time (per digit)</td>
                  <td className="py-2.5 px-3 text-right text-slate-600">
                    {base.batch_inference_ms_per_digit.toFixed(3)} ms
                  </td>
                  <td className="py-2.5 pl-3 text-right text-slate-900">
                    {enh.batch_inference_ms_per_digit.toFixed(3)} ms
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 flex flex-col justify-between gap-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">Live MNIST Benchmark Runner</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Execute live inference across ≥1,000 MNIST test images via <span className="font-mono">POST /api/evaluate</span>.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <label className="text-xs text-slate-600 flex items-center gap-2">
              <span>Test sample count</span>
              <select
                value={evalCount}
                onChange={(e) => setEvalCount(Number(e.target.value))}
                className="px-2.5 py-1.5 text-xs font-mono border border-slate-200 rounded-md bg-white text-slate-900"
              >
                <option value={1000}>1,000 images</option>
                <option value={2000}>2,000 images</option>
                <option value={5000}>5,000 images</option>
                <option value={10000}>10,000 images</option>
              </select>
            </label>

            <button
              type="button"
              onClick={() => onRunEvaluation(evalCount)}
              disabled={evalLoading}
              className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-sky-600 rounded-lg hover:bg-sky-700 disabled:opacity-50 cursor-pointer whitespace-nowrap shrink-0"
            >
              <Play className="w-3.5 h-3.5" />
              {evalLoading ? 'Evaluating...' : 'Run Benchmark'}
            </button>
          </div>

          {evalResult ? (
            <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-200 text-xs font-mono tabular-nums">
              <div>
                <div className="font-sans text-slate-500">Measured Accuracy</div>
                <div className="text-lg font-bold text-slate-900">
                  {(evalResult.accuracy * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <div className="font-sans text-slate-500">Macro F1-Score</div>
                <div className="text-lg font-bold text-slate-900">
                  {(evalResult.f1_macro * 100).toFixed(2)}%
                </div>
              </div>
              <div>
                <div className="font-sans text-slate-500">Samples Evaluated</div>
                <div className="text-sm font-semibold text-slate-800">
                  {evalResult.samples_evaluated.toLocaleString()}
                </div>
              </div>
              <div>
                <div className="font-sans text-slate-500">Total / Per-Digit Time</div>
                <div className="text-sm font-semibold text-slate-800">
                  {evalResult.total_inference_ms.toFixed(0)} ms ({evalResult.ms_per_digit.toFixed(3)} ms)
                </div>
              </div>
            </div>
          ) : (
            <div className="pt-3 border-t border-slate-200 text-xs text-slate-500">
              Click Run Benchmark to test the loaded model live on MNIST test samples and update the confusion matrix below.
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-5 flex flex-col gap-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Training &amp; Validation Curves</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Accuracy and cross-entropy loss across {epochs} training epochs with data augmentation.
            </p>
          </div>

          <div className="space-y-4">
            <div className="border border-slate-200 rounded-lg p-3 bg-slate-50">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-medium text-slate-700">Accuracy</span>
                <span className="text-slate-500 font-mono">
                  <span className="text-sky-600">Train</span> · <span className="text-emerald-700">Val</span>
                </span>
              </div>
              <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-44">
                {[0, 0.5, 1].map((t) => {
                  const y = padT + t * innerH;
                  const val = accMin + (1 - t) * (1 - accMin);
                  return (
                    <g key={t}>
                      <line x1={padL} x2={chartW - padR} y1={y} y2={y} stroke="#e2e8f0" strokeWidth="1" />
                      <text x={4} y={y + 3} fontSize="10" fill="#64748b" fontFamily="monospace">
                        {(val * 100).toFixed(1)}%
                      </text>
                    </g>
                  );
                })}
                <polyline
                  fill="none"
                  stroke="#0284c7"
                  strokeWidth="2.2"
                  points={toPolyline(enh.history.accuracy, accMin, 1.0)}
                />
                <polyline
                  fill="none"
                  stroke="#047857"
                  strokeWidth="2.2"
                  points={toPolyline(enh.history.val_accuracy, accMin, 1.0)}
                />
              </svg>
            </div>

            <div className="border border-slate-200 rounded-lg p-3 bg-slate-50">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-medium text-slate-700">Loss (Sparse Categorical Cross-Entropy)</span>
                <span className="text-slate-500 font-mono">
                  <span className="text-sky-600">Train</span> · <span className="text-emerald-700">Val</span>
                </span>
              </div>
              <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-44">
                {[0, 0.5, 1].map((t) => {
                  const y = padT + t * innerH;
                  const val = (1 - t) * lossMax;
                  return (
                    <g key={t}>
                      <line x1={padL} x2={chartW - padR} y1={y} y2={y} stroke="#e2e8f0" strokeWidth="1" />
                      <text x={4} y={y + 3} fontSize="10" fill="#64748b" fontFamily="monospace">
                        {val.toFixed(2)}
                      </text>
                    </g>
                  );
                })}
                <polyline
                  fill="none"
                  stroke="#0284c7"
                  strokeWidth="2.2"
                  points={toPolyline(enh.history.loss, 0, lossMax)}
                />
                <polyline
                  fill="none"
                  stroke="#047857"
                  strokeWidth="2.2"
                  points={toPolyline(enh.history.val_loss, 0, lossMax)}
                />
              </svg>
            </div>
          </div>
        </div>

        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-5 flex flex-col gap-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">Confusion Matrix (Digits 0–9)</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Rows = true labels, Columns = predicted labels ({evalResult ? `${evalResult.samples_evaluated} live samples` : '10,000 test images'}).
            </p>
          </div>

          <div className="overflow-x-auto">
            <div className="min-w-[420px]">
              <div className="grid grid-cols-11 gap-1 text-[11px] font-mono tabular-nums text-center mb-1">
                <div className="text-slate-400">T \ P</div>
                {Array.from({ length: 10 }, (_, d) => (
                  <div key={d} className="font-semibold text-slate-600">
                    {d}
                  </div>
                ))}
              </div>
              {displayCm.map((row, rIdx) => (
                <div key={rIdx} className="grid grid-cols-11 gap-1 text-[11px] font-mono tabular-nums items-center mb-1">
                  <div className="font-semibold text-slate-600 text-center">{rIdx}</div>
                  {row.map((val, cIdx) => {
                    const isDiag = rIdx === cIdx;
                    const intensity = isDiag ? 0.15 + 0.85 * (val / maxCmVal) : Math.min(0.8, val / 12);
                    const bg = isDiag
                      ? `rgba(2, 132, 199, ${intensity.toFixed(2)})`
                      : val > 0
                      ? `rgba(217, 119, 6, ${(0.12 + intensity * 0.55).toFixed(2)})`
                      : '#f8fafc';
                    const textColor = isDiag && intensity > 0.5 ? '#ffffff' : '#0f172a';
                    return (
                      <div
                        key={cIdx}
                        style={{ backgroundColor: bg, color: textColor }}
                        className="h-7 flex items-center justify-center rounded border border-slate-100"
                        title={`True ${rIdx} → Predicted ${cIdx}: ${val}`}
                      >
                        {val}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-slate-900 mb-1">Per-Digit Classification Report</h3>
          <p className="text-xs text-slate-500 mb-3">Precision, recall, and F1-score for classes 0–9.</p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-1.5 font-medium">Digit</th>
                  <th className="py-1.5 font-medium text-right">Precision</th>
                  <th className="py-1.5 font-medium text-right">Recall</th>
                  <th className="py-1.5 font-medium text-right">F1</th>
                  <th className="py-1.5 font-medium text-right">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                {Object.entries(per_class_metrics).map(([digit, m]) => (
                  <tr key={digit}>
                    <td className="py-1.5 font-semibold text-slate-900">{digit}</td>
                    <td className="py-1.5 text-right text-slate-700">{(m.precision * 100).toFixed(2)}%</td>
                    <td className="py-1.5 text-right text-slate-700">{(m.recall * 100).toFixed(2)}%</td>
                    <td className="py-1.5 text-right font-semibold text-sky-700">{(m.f1_score * 100).toFixed(2)}%</td>
                    <td className="py-1.5 text-right text-slate-500">{m.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 flex flex-col gap-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-900">CNN Architecture &amp; Sequence Limitations</h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Input: 28×28×1 grayscale · Adam optimizer · Sparse Categorical Cross-Entropy · 10-class Softmax.
            </p>
          </div>
          <div className="overflow-x-auto max-h-72">
            <table className="w-full text-xs text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500">
                  <th className="py-1.5 font-medium">Layer</th>
                  <th className="py-1.5 font-medium">Type</th>
                  <th className="py-1.5 font-medium">Output Shape</th>
                  <th className="py-1.5 font-medium text-right">Params</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono tabular-nums">
                {architecture_layers.map((layer, idx) => (
                  <tr key={idx}>
                    <td className="py-1.5 text-slate-800">{layer.name}</td>
                    <td className="py-1.5 text-slate-600 font-sans">{layer.type}</td>
                    <td className="py-1.5 text-slate-500">{layer.output_shape}</td>
                    <td className="py-1.5 text-right text-slate-700">{layer.params.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="border-t border-slate-200 pt-3 text-xs text-slate-600 leading-relaxed">
            <span className="font-semibold text-slate-900">Segmentation vs. End-to-End Sequence Recognition: </span>
            {modelInfo.sequence_note}
          </div>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-5">
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-slate-900">Generated Matplotlib Evaluation Figures</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            High-resolution figures saved directly by <span className="font-mono">training/train_cnn.py</span>.
          </p>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="border border-slate-200 rounded-lg p-3 bg-slate-50">
            <div className="text-xs font-medium text-slate-700 mb-2">Training &amp; Validation Curves</div>
            <img
              src={modelInfo.plots.training_curves}
              alt="Matplotlib training and validation curves"
              className="w-full h-auto rounded border border-slate-200 bg-white"
            />
          </div>
          <div className="border border-slate-200 rounded-lg p-3 bg-slate-50">
            <div className="text-xs font-medium text-slate-700 mb-2">MNIST Confusion Matrix Heatmap</div>
            <img
              src={modelInfo.plots.confusion_matrix}
              alt="Matplotlib MNIST confusion matrix"
              className="w-full h-auto rounded border border-slate-200 bg-white"
            />
          </div>
        </div>
      </div>
    </div>
  );
};
