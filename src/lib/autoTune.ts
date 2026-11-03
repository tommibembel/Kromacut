/**
 * Auto-tune: choose target colors, contrast expansion and Auto-paint settings
 * automatically from an image and the owned filaments.
 *
 * React-free and deterministic. Prototype stage (fork tommibembel/Kromacut).
 *
 * Pipeline:
 *  1. extractTargetColors   – weighted k-means in Lab, merge near-duplicates
 *                             (anti-aliased edges), keep small but distinct colors.
 *  2. expandContrast        – push target lightness apart around the weighted mean.
 *  3. autoTune              – run generateAutoLayers for a small grid of
 *                             (contrast, separation ΔE) candidates and score them
 *                             by fidelity to the ORIGINAL colors plus separation
 *                             between targets, with mild height penalties.
 */

import type { Filament } from '../types';
import { generateAutoLayers, type AutoPaintImageSwatch, type AutoPaintResult } from './autoPaint';
import { deltaE2000Lab, labToRgb, rgbToLab, type Lab } from './colorDifference';

export interface WeightedColor {
    hex: string;
    count: number;
}

export interface TargetColor {
    hex: string;
    lab: Lab;
    /** Normalized image weight 0..1 */
    weight: number;
}

export interface ExtractOptions {
    /** Upper bound for initial clusters. */
    maxClusters?: number;
    /** Clusters closer than this (ΔE00) are merged. */
    mergeDeltaE?: number;
    /** Clusters below this weight are dropped unless clearly distinct. */
    minWeight?: number;
    /** A small cluster survives when at least this far (ΔE00) from every bigger one. */
    distinctDeltaE?: number;
    /** Hard floor: anything below this weight is always dropped. */
    absoluteMinWeight?: number;
    /** Maximum number of target colors returned. */
    maxTargets?: number;
}

const hexOf = (rgb: [number, number, number]) =>
    '#' + rgb.map((c) => Math.round(c).toString(16).padStart(2, '0')).join('');

const parseHex = (hex: string): [number, number, number] => {
    const h = hex.replace('#', '');
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
};

const labDist2 = (a: Lab, b: Lab) => (a.L - b.L) ** 2 + (a.a - b.a) ** 2 + (a.b - b.b) ** 2;

/** Weighted k-means in Lab with deterministic farthest-point seeding. */
function kmeansLab(points: { lab: Lab; w: number }[], k: number, iterations = 25): { lab: Lab; w: number }[] {
    if (points.length <= k) return points.map((p) => ({ lab: { ...p.lab }, w: p.w }));
    const centers: Lab[] = [];
    // seed: heaviest point, then farthest weighted
    let heaviest = 0;
    for (let i = 1; i < points.length; i++) if (points[i].w > points[heaviest].w) heaviest = i;
    centers.push({ ...points[heaviest].lab });
    const minD = points.map((p) => labDist2(p.lab, centers[0]));
    while (centers.length < k) {
        let best = -1;
        let bestScore = -1;
        for (let i = 0; i < points.length; i++) {
            // sqrt(w) so small distinct colors can still seed a cluster
            const s = minD[i] * Math.sqrt(points[i].w);
            if (s > bestScore) {
                bestScore = s;
                best = i;
            }
        }
        if (best < 0 || bestScore <= 0) break;
        centers.push({ ...points[best].lab });
        for (let i = 0; i < points.length; i++) minD[i] = Math.min(minD[i], labDist2(points[i].lab, points[best].lab));
    }
    const assign = new Int32Array(points.length);
    const weights = new Float64Array(centers.length);
    for (let it = 0; it < iterations; it++) {
        let moved = false;
        for (let i = 0; i < points.length; i++) {
            let bi = 0;
            let bd = Infinity;
            for (let c = 0; c < centers.length; c++) {
                const d = labDist2(points[i].lab, centers[c]);
                if (d < bd) {
                    bd = d;
                    bi = c;
                }
            }
            if (assign[i] !== bi) moved = true;
            assign[i] = bi;
        }
        const sums = centers.map(() => ({ L: 0, a: 0, b: 0, w: 0 }));
        for (let i = 0; i < points.length; i++) {
            const s = sums[assign[i]];
            const { lab, w } = points[i];
            s.L += lab.L * w;
            s.a += lab.a * w;
            s.b += lab.b * w;
            s.w += w;
        }
        for (let c = 0; c < centers.length; c++) {
            const s = sums[c];
            weights[c] = s.w;
            if (s.w > 0) centers[c] = { L: s.L / s.w, a: s.a / s.w, b: s.b / s.w };
        }
        if (!moved && it > 0) break;
    }
    return centers.map((lab, c) => ({ lab, w: weights[c] })).filter((c) => c.w > 0);
}

export function extractTargetColors(colors: WeightedColor[], options: ExtractOptions = {}): TargetColor[] {
    const {
        maxClusters = 16,
        mergeDeltaE = 10,
        minWeight = 0.01,
        distinctDeltaE = 30,
        absoluteMinWeight = 0.00005,
        maxTargets = 8,
    } = options;
    const total = colors.reduce((s, c) => s + c.count, 0) || 1;
    const points = colors.map((c) => ({ lab: rgbToLab(parseHex(c.hex)), w: c.count / total }));
    let clusters = kmeansLab(points, maxClusters);

    // merge near-duplicates (heaviest first absorbs)
    clusters.sort((x, y) => y.w - x.w);
    const merged: { lab: Lab; w: number }[] = [];
    for (const c of clusters) {
        const host = merged.find((m) => deltaE2000Lab(m.lab, c.lab) < mergeDeltaE);
        if (host) {
            const w = host.w + c.w;
            host.lab = {
                L: (host.lab.L * host.w + c.lab.L * c.w) / w,
                a: (host.lab.a * host.w + c.lab.a * c.w) / w,
                b: (host.lab.b * host.w + c.lab.b * c.w) / w,
            };
            host.w = w;
        } else merged.push({ lab: { ...c.lab }, w: c.w });
    }
    merged.sort((x, y) => y.w - x.w);

    // keep big ones; keep small ones only if clearly distinct from all kept ones
    const kept: { lab: Lab; w: number }[] = [];
    for (const c of merged) {
        if (c.w < absoluteMinWeight) continue;
        const nearest = kept.reduce((m, k) => Math.min(m, deltaE2000Lab(k.lab, c.lab)), Infinity);
        if (c.w >= minWeight || nearest >= distinctDeltaE) kept.push(c);
        else {
            // fold its weight into the nearest kept color so totals stay meaningful
            let host = kept[0];
            for (const k of kept) if (deltaE2000Lab(k.lab, c.lab) < deltaE2000Lab(host.lab, c.lab)) host = k;
            if (host) host.w += c.w;
        }
    }
    kept.sort((x, y) => y.w - x.w);
    const out = kept.slice(0, maxTargets);
    const sum = out.reduce((s, c) => s + c.w, 0) || 1;
    return out.map((c) => {
        const rgb = labToRgb(c.lab) as [number, number, number];
        return { hex: hexOf(rgb), lab: rgbToLab(rgb), weight: c.w / sum };
    });
}

/**
 * Push lightness apart around the middle of the lightness range; strength 0 = unchanged.
 * The unweighted range midpoint (not the weighted mean) is used so a dominant
 * background also moves away from small foreground colors.
 */
export function expandContrast(targets: TargetColor[], strength: number): TargetColor[] {
    if (strength <= 0) return targets;
    const Ls = targets.map((t) => t.lab.L);
    const mean = (Math.min(...Ls) + Math.max(...Ls)) / 2;
    return targets.map((t) => {
        const L = Math.max(0, Math.min(100, mean + (t.lab.L - mean) * (1 + strength)));
        const rgb = labToRgb({ L, a: t.lab.a, b: t.lab.b }) as [number, number, number];
        return { hex: hexOf(rgb), lab: rgbToLab(rgb), weight: t.weight };
    });
}

export interface AutoTuneOptions {
    layerHeight: number;
    firstLayerHeight: number;
    maxHeight?: number;
    contrastStrengths?: number[];
    separationDeltaEs?: number[];
    /** 0 = only fidelity, 1 = strongly prefer separation between targets. */
    contrastWeight?: number;
    extract?: ExtractOptions;
    algorithm?: 'fast' | 'balanced' | 'thorough';
}

export interface AutoTuneCandidate {
    contrastStrength: number;
    separationDeltaE: number;
    targets: TargetColor[];
    predicted: string[];
    score: number;
    fidelity: number; // weighted ΔE00 predicted vs original
    worstFidelity: number;
    separationPenalty: number;
    separated: boolean;
    totalHeight: number;
    filamentOrder: string[];
    result: AutoPaintResult;
}

export interface AutoTuneResult {
    originalTargets: TargetColor[];
    best: AutoTuneCandidate;
    candidates: Omit<AutoTuneCandidate, 'result'>[];
    /** Original target colors the best stack misses by more than ΔE00 20. */
    unreachable: { hex: string; weight: number; deltaE: number }[];
}

export function autoTune(
    image: WeightedColor[],
    filaments: Filament[],
    options: AutoTuneOptions
): AutoTuneResult {
    const {
        layerHeight,
        firstLayerHeight,
        maxHeight,
        contrastStrengths = [0, 0.2, 0.4, 0.6],
        separationDeltaEs = [6, 10, 15],
        contrastWeight = 0.6,
        algorithm = 'balanced',
    } = options;
    const original = extractTargetColors(image, options.extract);
    const candidates: AutoTuneCandidate[] = [];

    for (const strength of contrastStrengths) {
        const targets = expandContrast(original, strength);
        const swatches: AutoPaintImageSwatch[] = targets.map((t) => ({
            hex: t.hex,
            count: Math.max(1, Math.round(t.weight * 1e6)),
        }));
        for (const de of separationDeltaEs) {
            let result: AutoPaintResult;
            try {
                result = generateAutoLayers(filaments, swatches, layerHeight, firstLayerHeight, maxHeight, true, false, {
                    algorithm,
                    maxExtraRepeats: 0,
                    preserveSeparation: true,
                    separationMaxDeltaE: de,
                    failOnSeparationError: false,
                });
            } catch {
                continue;
            }
            const mappings = result.finalStack.targetMappings;
            if (mappings.length !== targets.length) continue;
            const pred = mappings.map((m) => ({ L: m.predictedLab[0], a: m.predictedLab[1], b: m.predictedLab[2] }));
            // fidelity against ORIGINAL colors
            let fidelity = 0;
            let worst = 0;
            original.forEach((o, i) => {
                const d = deltaE2000Lab(o.lab, pred[i]);
                fidelity += d * o.weight;
                worst = Math.max(worst, d);
            });
            // separation: predicted pairs should stay at least as far apart as the
            // originals (capped at 25), weighted by the smaller region
            let separationPenalty = 0;
            for (let i = 0; i < original.length; i++)
                for (let j = i + 1; j < original.length; j++) {
                    const want = Math.min(25, deltaE2000Lab(original[i].lab, original[j].lab));
                    const got = deltaE2000Lab(pred[i], pred[j]);
                    const pairWeight = Math.sqrt(Math.min(original[i].weight, original[j].weight));
                    separationPenalty += Math.max(0, want - got) * pairWeight;
                    // printed layers tend to wash out lightness differences; ask for
                    // at least the original lightness gap between the two regions
                    const wantL = Math.abs(original[i].lab.L - original[j].lab.L);
                    const gotL = Math.abs(pred[i].L - pred[j].L);
                    separationPenalty += Math.max(0, wantL - gotL) * pairWeight;
                }
            const separated = result.colorSeparation?.satisfied ?? false;
            const score =
                (1 - contrastWeight) * fidelity +
                contrastWeight * separationPenalty * 2 +
                (separated ? 0 : 5) +
                result.totalHeight * 0.5;
            candidates.push({
                contrastStrength: strength,
                separationDeltaE: de,
                targets,
                predicted: mappings.map((m) => m.predictedColor.hex),
                score,
                fidelity,
                worstFidelity: worst,
                separationPenalty,
                separated,
                totalHeight: result.totalHeight,
                filamentOrder: result.filamentOrder,
                result,
            });
        }
    }
    if (candidates.length === 0) throw new Error('autoTune: no Auto-paint candidate could be generated');
    candidates.sort((x, y) => x.score - y.score);
    const best = candidates[0];
    const unreachable = original
        .map((o, i) => ({
            hex: o.hex,
            weight: o.weight,
            deltaE: deltaE2000Lab(o.lab, {
                L: best.result.finalStack.targetMappings[i].predictedLab[0],
                a: best.result.finalStack.targetMappings[i].predictedLab[1],
                b: best.result.finalStack.targetMappings[i].predictedLab[2],
            }),
        }))
        .filter((u) => u.deltaE > 20);
    return {
        originalTargets: original,
        best,
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        candidates: candidates.map(({ result, ...rest }) => rest),
        unreachable,
    };
}
