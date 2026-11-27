import React from 'react';
import { Trash2 } from 'lucide-react';
import { HistoryEntry, SampleItem } from '../types';

interface SampleGalleryAndHistoryProps {
  samples: SampleItem[];
  history: HistoryEntry[];
  onSelectSample: (sample: SampleItem) => void;
  onClearHistory: () => void;
  isLoading: boolean;
}

export const SampleGalleryAndHistory: React.FC<SampleGalleryAndHistoryProps> = ({
  samples,
  history,
  onSelectSample,
  onClearHistory,
  isLoading,
}) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <section id="samples" className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-5 flex flex-col gap-4">
        <div>
          <h2 className="text-base font-semibold text-slate-900">Sample Handwriting Gallery</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Click any single, double, or multi-digit test image (light paper &amp; dark board) to run recognition.
          </p>
        </div>

        {samples.length === 0 ? (
          <div className="text-xs text-slate-500 py-8 text-center border border-slate-200 rounded-lg bg-slate-50">
            Loading sample handwriting images...
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {samples.map((sample) => (
              <button
                key={sample.id}
                type="button"
                disabled={isLoading}
                onClick={() => onSelectSample(sample)}
                className="group flex flex-col gap-2 p-2.5 rounded-lg border border-slate-200 bg-white hover:border-sky-500 text-left transition-colors cursor-pointer disabled:opacity-50"
              >
                <div className="h-20 w-full rounded border border-slate-200 overflow-hidden flex items-center justify-center bg-slate-100">
                  <img
                    src={sample.image_base64}
                    alt={sample.label}
                    className="max-h-full max-w-full object-contain"
                  />
                </div>
                <div>
                  <div className="text-xs font-semibold text-slate-900 group-hover:text-sky-700 truncate">
                    {sample.label}
                  </div>
                  <div className="text-[11px] text-slate-500 font-mono">
                    {sample.mode} · {sample.background} bg
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </section>

      <section className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-5 flex flex-col gap-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-slate-900">Prediction History</h2>
            <p className="text-xs text-slate-500 mt-0.5">Recent predictions with timestamps and confidence.</p>
          </div>
          {history.length > 0 && (
            <button
              type="button"
              onClick={onClearHistory}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium text-rose-700 hover:bg-rose-50 rounded-md transition-colors cursor-pointer whitespace-nowrap shrink-0"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear History
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <div className="flex-1 flex items-center justify-center py-10 text-xs text-slate-500 border border-slate-200 rounded-lg bg-slate-50">
            No predictions recorded yet. Draw or upload an image to begin.
          </div>
        ) : (
          <div className="divide-y divide-slate-100 max-h-72 overflow-y-auto pr-1">
            {history.map((item) => (
              <div key={item.id} className="py-2.5 flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <img
                    src={item.processed_preview_base64}
                    alt={`Prediction ${item.predicted_number}`}
                    className="w-12 h-9 object-contain rounded border border-slate-200 bg-slate-950 shrink-0"
                  />
                  <div className="min-w-0">
                    <div className="text-sm font-bold font-mono tabular-nums text-slate-900 truncate">
                      {item.predicted_number}
                    </div>
                    <div className="text-[11px] text-slate-500 font-mono tabular-nums truncate">
                      {item.mode} · {item.source} · {item.timestamp}
                    </div>
                  </div>
                </div>
                <div className="text-right shrink-0 font-mono tabular-nums">
                  <div
                    className={`text-xs font-semibold ${
                      item.low_confidence_warning || item.segmentation_uncertain
                        ? 'text-amber-700'
                        : 'text-emerald-700'
                    }`}
                  >
                    {(item.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-[11px] text-slate-400">{item.processing_time_ms.toFixed(1)} ms</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};
