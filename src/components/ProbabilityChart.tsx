import React from 'react';

interface ProbabilityChartProps {
  probabilities: number[];
  predictedDigit?: number;
  title?: string;
}

export const ProbabilityChart: React.FC<ProbabilityChartProps> = ({
  probabilities,
  predictedDigit,
  title = 'Softmax Probability Distribution (Digits 0–9)',
}) => {
  const safeProbs = probabilities.length === 10 ? probabilities : Array(10).fill(0);
  const maxVal = Math.max(...safeProbs, 0.01);
  const topIdx = safeProbs.indexOf(Math.max(...safeProbs));
  const activeDigit = predictedDigit !== undefined ? predictedDigit : topIdx;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-slate-700">{title}</span>
        <span className="text-xs font-mono tabular-nums text-slate-500">
          Top class: {activeDigit} ({(safeProbs[activeDigit] * 100).toFixed(2)}%)
        </span>
      </div>

      <div className="grid grid-cols-10 gap-2 items-end h-36 pt-5 pb-1 px-2 bg-slate-50 border border-slate-200 rounded-lg">
        {safeProbs.map((prob, digit) => {
          const pct = Math.max(0, Math.min(100, prob * 100));
          const barHeight = Math.max(4, Math.round((prob / Math.max(maxVal, 1)) * 88));
          const isWinner = digit === activeDigit && prob > 0;
          return (
            <div key={digit} className="flex flex-col items-center justify-end h-full gap-1.5">
              <span
                className={`text-[10px] font-mono tabular-nums ${
                  isWinner ? 'text-sky-700 font-semibold' : 'text-slate-400'
                }`}
              >
                {pct >= 1 ? `${pct.toFixed(0)}%` : pct > 0.05 ? `${pct.toFixed(1)}` : '0'}
              </span>
              <div className="w-full flex items-end justify-center h-[88px]">
                <div
                  className={`w-full max-w-[26px] rounded-t transition-all duration-200 ${
                    isWinner ? 'bg-sky-600' : 'bg-slate-300'
                  }`}
                  style={{ height: `${barHeight}px` }}
                  title={`Digit ${digit}: ${(prob * 100).toFixed(2)}%`}
                />
              </div>
              <span
                className={`text-xs font-mono tabular-nums ${
                  isWinner ? 'text-slate-900 font-bold' : 'text-slate-500'
                }`}
              >
                {digit}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
