export const SAMPLE_RATE = 16000;
export const WIN_LENGTH = 400;
export const HOP_LENGTH = 160;
export const CAUSAL_LEFT_PAD = WIN_LENGTH - HOP_LENGTH;
export const N_FFT = 512;
export const N_FREQ_BINS = N_FFT / 2 + 1;
export const N_MELS = 80;
export const F_MIN = 64;
export const F_MAX = 7800;
export const MEL_EPS = 1e-6;
export const DEFAULT_MEL_MEAN = -6.0;
export const DEFAULT_MEL_STD = 4.0;
export const DEFAULT_MEL_VAR = 4.0;
export const DEFAULT_EMA_DECAY = 0.02;
export const OUTPUT_HZ = 50;
export const FRAME_DURATION_SEC = 1 / OUTPUT_HZ;
export const HIDDEN_STATE_SHAPE = [1, 1, 1] as const;
export const HIDDEN_STATE_SIZE = 1;
export const CONV_CACHE_SHAPE = [1, 288, 248] as const;
export const CONV_CACHE_SIZE = 1 * 288 * 248;

export const OCULUS_VISEMES = [
  "sil",
  "PP",
  "FF",
  "TH",
  "DD",
  "kk",
  "CH",
  "SS",
  "nn",
  "RR",
  "aa",
  "E",
  "ih",
  "oh",
  "ou",
] as const;

export type OculusViseme = (typeof OCULUS_VISEMES)[number];

export const RHUBARB_SHAPES = ["X", "A", "B", "C", "D", "E", "F", "G", "H"] as const;

export type RhubarbShape = (typeof RHUBARB_SHAPES)[number];

export const MOUTH_CURVE_NAMES = ["jawOpen", "lipRound", "lipClose"] as const;

export type MouthCurveName = (typeof MOUTH_CURVE_NAMES)[number];

export interface MouthCurves {
  jawOpen: number;
  lipRound: number;
  lipClose: number;
}

export interface FramePrediction {
  time: number;
  viseme15Probs: Float32Array;
  rhubarb9Probs: Float32Array;
  viseme15: OculusViseme;
  rhubarb9: RhubarbShape;
  mouthCurves: MouthCurves;
}

export interface LipSyncCue<T extends string = string> {
  start: number;
  end: number;
  value: T;
  confidence: number;
}

export interface RhubarbMetadata {
  soundFile: string;
  duration: number;
}

export interface RhubarbJsonOutput {
  metadata: RhubarbMetadata;
  mouthCues: Array<{
    start: number;
    end: number;
    value: RhubarbShape;
  }>;
}

export const RHUBARB_DESCRIPTIONS: Record<RhubarbShape, { label: string; phonemes: string }> = {
  X: { label: "Rest / Silence", phonemes: "rest" },
  A: { label: "Closed Lips", phonemes: "p, b, m" },
  B: { label: "Slightly Open / Teeth", phonemes: "k, s, t, d, n, i" },
  C: { label: "Open Mid Vowel", phonemes: "e, eh, ae" },
  D: { label: "Wide Open", phonemes: "a, aa, ah" },
  E: { label: "Rounded Open", phonemes: "o, oh, r" },
  F: { label: "Puckered / Narrow", phonemes: "u, oo, w" },
  G: { label: "Upper Teeth on Lip", phonemes: "f, v" },
  H: { label: "Tongue Behind Teeth", phonemes: "l" },
};

export const VISEME15_TO_RHUBARB9_IDX: readonly number[] = [
  0, // sil -> X
  1, // PP  -> A
  7, // FF  -> G
  2, // TH  -> B
  2, // DD  -> B
  2, // kk  -> B
  2, // CH  -> B
  2, // SS  -> B
  2, // nn  -> B
  5, // RR  -> E
  4, // aa  -> D
  3, // E   -> C
  2, // ih  -> B
  5, // oh  -> E
  6, // ou  -> F
];

export const VISEME15_MOUTH_CURVES: ReadonlyArray<readonly [number, number, number]> = [
  [0.0, 0.0, 1.0],   // sil
  [0.0, 0.0, 1.0],   // PP
  [0.1, 0.0, 0.85],  // FF
  [0.2, 0.0, 0.35],  // TH
  [0.22, 0.0, 0.3],  // DD
  [0.28, 0.0, 0.25], // kk
  [0.25, 0.35, 0.3], // CH
  [0.16, 0.0, 0.45], // SS
  [0.24, 0.0, 0.25], // nn
  [0.32, 0.45, 0.2], // RR
  [0.95, 0.05, 0.0], // aa
  [0.62, 0.05, 0.05],// E
  [0.35, 0.0, 0.15], // ih
  [0.65, 0.75, 0.05],// oh
  [0.32, 0.95, 0.15],// ou
];
