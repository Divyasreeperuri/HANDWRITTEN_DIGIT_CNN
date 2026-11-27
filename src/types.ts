export type RecognitionMode = 'single' | 'double' | 'multi';

export interface DigitProbability {
  digit: number;
  probability: number;
}

export interface SegmentedDigitPrediction {
  index: number;
  digit: number;
  confidence: number;
  probabilities: number[];
  top_alternatives: DigitProbability[];
  bbox: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  was_split: boolean;
  processed_digit_base64: string;
}

export interface PredictionResponse {
  predicted_number: string;
  mode: RecognitionMode;
  confidence: number;
  min_confidence: number;
  digit_count: number;
  digits: SegmentedDigitPrediction[];
  primary_probabilities: number[];
  low_confidence_warning: boolean;
  segmentation_uncertain: boolean;
  warnings: string[];
  inverted_colors: boolean;
  processed_preview_base64: string;
  binary_preview_base64: string;
  processing_time_ms: number;
}

export interface SampleItem {
  id: string;
  label: string;
  expected: string;
  mode: RecognitionMode;
  background: 'white' | 'dark';
  image_url: string;
  image_base64: string;
}

export interface HistoryEntry {
  id: string;
  timestamp: string;
  predicted_number: string;
  mode: RecognitionMode;
  confidence: number;
  digit_count: number;
  processing_time_ms: number;
  source: 'canvas' | 'upload' | 'sample';
  processed_preview_base64: string;
  low_confidence_warning: boolean;
  segmentation_uncertain: boolean;
}

export interface ModelMetricsPayload {
  model_name: string;
  dataset: string;
  input_shape: number[];
  num_classes: number;
  parameters: number;
  test_samples: number;
  enhanced_cnn: {
    test_accuracy: number;
    test_loss: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
    training_seconds: number;
    batch_inference_ms_per_digit: number;
    single_inference_ms: number;
    epochs_run: number;
    history: {
      accuracy: number[];
      val_accuracy: number[];
      loss: number[];
      val_loss: number[];
    };
  };
  baseline_cnn: {
    test_accuracy: number;
    test_loss: number;
    precision_macro: number;
    recall_macro: number;
    f1_macro: number;
    training_seconds: number;
    batch_inference_ms_per_digit: number;
    parameters: number;
    epochs_run: number;
    history: {
      accuracy: number[];
      val_accuracy: number[];
      loss: number[];
      val_loss: number[];
    };
  };
  confusion_matrix: number[][];
  per_class_metrics: Record<
    string,
    {
      precision: number;
      recall: number;
      f1_score: number;
      support: number;
    }
  >;
  architecture_layers: {
    name: string;
    type: string;
    output_shape: string;
    params: number;
  }[];
}

export interface HealthResponse {
  status: string;
  ready: boolean;
  model_loaded: boolean;
  model_state: 'loading' | 'training' | 'ready' | 'error';
  message: string;
  model_name: string;
  model_format?: string | null;
  model_path?: string;
  test_accuracy?: number | null;
  runtime?: string;
}

export interface ModelInfoResponse {
  status: string;
  ready?: boolean;
  model_state?: 'loading' | 'training' | 'ready' | 'error';
  message?: string;
  model_format?: string | null;
  model_path: string;
  metrics: ModelMetricsPayload | null;
  plots: {
    training_curves: string;
    confusion_matrix: string;
  };
  sequence_note: string;
}

export interface EvaluationResponse {
  samples_evaluated: number;
  accuracy: number;
  loss: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  total_inference_ms: number;
  ms_per_digit: number;
  confusion_matrix: number[][];
}
