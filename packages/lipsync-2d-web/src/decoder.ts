import {
  FRAME_DURATION_SEC,
  OCULUS_VISEMES,
  RHUBARB_SHAPES,
  type LipSyncCue,
  type OculusViseme,
  type RhubarbShape,
} from "./constants.js";

export interface ViterbiDecodeOptions {
  frameDurationSec?: number;
  switchPenalty?: number;
  silenceIndex?: number;
  enterSilenceThreshold?: number;
  exitSilenceThreshold?: number;
  minDurationFrames?: number;
  protectedClasses?: readonly number[];
  protectedMinFrames?: number;
  audioRmsDb?: Float32Array;
  rmsSilenceDb?: number;
  totalDurationSec?: number;
}

interface FrameRun {
  startFrame: number;
  endFrame: number;
  classIdx: number;
}

function roundTime(sec: number): number {
  return Math.round(sec * 1000) / 1000;
}

function mergeShortRuns(
  runs: FrameRun[],
  probs: Float32Array,
  numClasses: number,
  minDurationFrames: number,
  protectedClasses: ReadonlySet<number>,
  protectedMinFrames: number,
): FrameRun[] {
  if (runs.length <= 1 || (minDurationFrames <= 1 && protectedMinFrames <= 1)) {
    return runs;
  }

  const working = runs.map((r) => ({ ...r }));

  while (working.length > 1) {
    let shortestIdx = -1;
    let shortestLen = Infinity;
    for (let i = 0; i < working.length; i++) {
      const run = working[i]!;
      const req = protectedClasses.has(run.classIdx) ? protectedMinFrames : minDurationFrames;
      const len = run.endFrame - run.startFrame;
      if (len < req && len < shortestLen) {
        shortestLen = len;
        shortestIdx = i;
      }
    }

    if (shortestIdx === -1) {
      break;
    }

    const target = working[shortestIdx]!;
    const left = shortestIdx > 0 ? working[shortestIdx - 1] : undefined;
    const right = shortestIdx < working.length - 1 ? working[shortestIdx + 1] : undefined;

    let mergeLeft = false;
    if (left && right) {
      let leftScore = 0;
      let rightScore = 0;
      for (let t = target.startFrame; t < target.endFrame; t++) {
        leftScore += probs[t * numClasses + left.classIdx] ?? 0;
        rightScore += probs[t * numClasses + right.classIdx] ?? 0;
      }
      mergeLeft = leftScore >= rightScore;
    } else if (left) {
      mergeLeft = true;
    }

    if (mergeLeft && left) {
      left.endFrame = target.endFrame;
      working.splice(shortestIdx, 1);
    } else if (right) {
      right.startFrame = target.startFrame;
      working.splice(shortestIdx, 1);
    } else {
      break;
    }

    for (let i = working.length - 1; i > 0; i--) {
      if (working[i]!.classIdx === working[i - 1]!.classIdx) {
        working[i - 1]!.endFrame = working[i]!.endFrame;
        working.splice(i, 1);
      }
    }
  }

  return working;
}

/**
 * Decode 50 Hz frame class probabilities into clean, non-overlapping cues using
 * silence-gate hysteresis, log-space Viterbi DP, and minimum-duration merging.
 */
export function decodeViterbiCues<T extends string>(
  flatProbs: Float32Array,
  labels: readonly T[],
  options: ViterbiDecodeOptions = {},
): LipSyncCue<T>[] {
  const numClasses = labels.length;
  if (numClasses === 0 || flatProbs.length === 0) {
    return [];
  }
  const numFrames = Math.floor(flatProbs.length / numClasses);
  if (numFrames === 0) {
    return [];
  }

  const frameDuration = options.frameDurationSec ?? FRAME_DURATION_SEC;
  const switchPenalty = options.switchPenalty ?? 1.2;
  const silenceIndex = options.silenceIndex ?? 0;
  const enterSilenceThreshold = options.enterSilenceThreshold ?? 0.65;
  const exitSilenceThreshold = options.exitSilenceThreshold ?? 0.45;
  const minDurationFrames = Math.max(1, options.minDurationFrames ?? 2);
  const protectedClasses = new Set(options.protectedClasses ?? [1]);
  const protectedMinFrames = Math.max(1, options.protectedMinFrames ?? 1);
  const audioRmsDb = options.audioRmsDb;
  const rmsSilenceDb = options.rmsSilenceDb ?? -52.0;

  // Silence-gate hysteresis generates per-frame log-likelihoods matching Python decode.py.
  const logEmit = new Float64Array(numFrames * numClasses);
  let inSilence = true;

  for (let t = 0; t < numFrames; t++) {
    const base = t * numClasses;
    const pSil = flatProbs[base + silenceIndex] ?? 0;
    const pSpeech = 1.0 - pSil;
    const rmsQuiet =
      audioRmsDb !== undefined && t < audioRmsDb.length && audioRmsDb[t]! < rmsSilenceDb;

    if (inSilence) {
      if (!rmsQuiet && pSpeech >= exitSilenceThreshold) {
        inSilence = false;
      }
    } else if (rmsQuiet || pSil >= enterSilenceThreshold) {
      inSilence = true;
    }

    for (let c = 0; c < numClasses; c++) {
      const p = Math.min(1.0, Math.max(flatProbs[base + c] ?? 0, 1e-6));
      let lp = Math.log(p);
      if (inSilence) {
        lp = c === silenceIndex ? 0.0 : -12.0;
      } else if (c === silenceIndex) {
        lp = Math.min(lp, -4.0);
      }
      logEmit[base + c] = lp;
    }
  }

  const backptr = new Uint8Array(numFrames * numClasses);
  let prevScores = new Float64Array(numClasses);
  let currScores = new Float64Array(numClasses);

  for (let c = 0; c < numClasses; c++) {
    prevScores[c] = logEmit[c]!;
  }

  for (let t = 1; t < numFrames; t++) {
    const base = t * numClasses;

    let bestPrevIdx = 0;
    let bestPrevScore = prevScores[0]!;
    for (let c = 1; c < numClasses; c++) {
      if (prevScores[c]! > bestPrevScore) {
        bestPrevScore = prevScores[c]!;
        bestPrevIdx = c;
      }
    }

    for (let c = 0; c < numClasses; c++) {
      const stayScore = prevScores[c]!;
      const switchScore = bestPrevScore - switchPenalty;
      if (stayScore >= switchScore) {
        currScores[c] = logEmit[base + c]! + stayScore;
        backptr[base + c] = c;
      } else {
        currScores[c] = logEmit[base + c]! + switchScore;
        backptr[base + c] = bestPrevIdx;
      }
    }

    const tmp = prevScores;
    prevScores = currScores;
    currScores = tmp;
  }

  const path = new Int32Array(numFrames);
  let lastClass = 0;
  let bestFinal = prevScores[0]!;
  for (let c = 1; c < numClasses; c++) {
    if (prevScores[c]! > bestFinal) {
      bestFinal = prevScores[c]!;
      lastClass = c;
    }
  }
  path[numFrames - 1] = lastClass;
  for (let t = numFrames - 1; t > 0; t--) {
    path[t - 1] = backptr[t * numClasses + path[t]!]!;
  }

  const rawRuns: FrameRun[] = [];
  let runStart = 0;
  let runClass = path[0]!;
  for (let t = 1; t < numFrames; t++) {
    if (path[t]! !== runClass) {
      rawRuns.push({ startFrame: runStart, endFrame: t, classIdx: runClass });
      runStart = t;
      runClass = path[t]!;
    }
  }
  rawRuns.push({ startFrame: runStart, endFrame: numFrames, classIdx: runClass });

  const merged = mergeShortRuns(
    rawRuns,
    flatProbs,
    numClasses,
    minDurationFrames,
    protectedClasses,
    protectedMinFrames,
  );
  const totalDuration = options.totalDurationSec ?? numFrames * frameDuration;

  const cues: LipSyncCue<T>[] = [];
  for (let i = 0; i < merged.length; i++) {
    const run = merged[i]!;
    const start = i === 0 ? 0 : roundTime(run.startFrame * frameDuration);
    const end =
      i === merged.length - 1
        ? Math.max(start, roundTime(totalDuration))
        : roundTime(run.endFrame * frameDuration);

    let confSum = 0;
    const len = Math.max(1, run.endFrame - run.startFrame);
    for (let t = run.startFrame; t < run.endFrame; t++) {
      confSum += flatProbs[t * numClasses + run.classIdx] ?? 0;
    }

    cues.push({
      start,
      end,
      value: labels[run.classIdx]!,
      confidence: Math.round((confSum / len) * 1000) / 1000,
    });
  }

  return cues;
}

export function decodeRhubarbCues(
  rhubarbProbs: Float32Array,
  options?: ViterbiDecodeOptions,
): LipSyncCue<RhubarbShape>[] {
  return decodeViterbiCues(rhubarbProbs, RHUBARB_SHAPES, options);
}

export function decodeVisemeCues(
  visemeProbs: Float32Array,
  options?: ViterbiDecodeOptions,
): LipSyncCue<OculusViseme>[] {
  return decodeViterbiCues(visemeProbs, OCULUS_VISEMES, options);
}
