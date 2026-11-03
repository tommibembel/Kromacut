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

/** Downscaled image pixels, RGB or RGBA, row-major. */
export interface ImagePixels {
    width: number;
    height: number;
    data: ArrayLike<number>;
    channels?: 3 | 4;
}

const binKey = (r: number, g: number, b: number) => ((r >> 2) << 12) | ((g >> 2) << 6) | (b >> 2);
const binRgb = (k: number): [number, number, number] => [((k >> 12) & 63) << 2, ((k >> 6) & 63) << 2, (k & 63) << 2];

interface Group {
    lab: Lab;
    w: number;
    members: number[]; // original cluster indices
}

/**
 * Find the target colors of an image.
 *
 * With pixel input the extractor is edge-aware: two color groups that meet
 * along a sharp edge (text on background, outlines) are never merged, and a
 * small group survives when it forms a connected blob with a sharp boundary
 * (a beard, a thin stripe) instead of scattered noise.
 */
export function extractTargetColors(
    input: WeightedColor[] | ImagePixels,
    options: ExtractOptions = {}
): TargetColor[] {
    const {
        maxClusters = 16,
        mergeDeltaE = 10,
        minWeight = 0.01,
        distinctDeltaE = 30,
        absoluteMinWeight = 0.00005,
        maxTargets = 8,
    } = options;

    // ---- histogram -------------------------------------------------------
    const pixels = Array.isArray(input) ? null : (input as ImagePixels);
    let colors: WeightedColor[];
    let pixelBins: Int32Array | null = null;
    if (pixels) {
        const ch = pixels.channels ?? (pixels.data.length === pixels.width * pixels.height * 4 ? 4 : 3);
        const n = pixels.width * pixels.height;
        pixelBins = new Int32Array(n);
        const counts = new Map<number, number>();
        for (let i = 0; i < n; i++) {
            const o = i * ch;
            if (ch === 4 && pixels.data[o + 3] < 128) {
                pixelBins[i] = -1;
                continue;
            }
            const k = binKey(pixels.data[o], pixels.data[o + 1], pixels.data[o + 2]);
            pixelBins[i] = k;
            counts.set(k, (counts.get(k) ?? 0) + 1);
        }
        colors = [...counts].map(([k, count]) => ({ hex: hexOf(binRgb(k)), count }));
    } else colors = input as WeightedColor[];

    const total = colors.reduce((s, c) => s + c.count, 0) || 1;
    const points = colors.map((c) => ({ lab: rgbToLab(parseHex(c.hex)), w: c.count / total }));
    const clusters = kmeansLab(points, maxClusters);
    const nC = clusters.length;
    const nearestCluster = (lab: Lab) => {
        let bi = 0;
        let bd = Infinity;
        for (let c = 0; c < nC; c++) {
            const d = labDist2(lab, clusters[c].lab);
            if (d < bd) {
                bd = d;
                bi = c;
            }
        }
        return bi;
    };

    // ---- spatial statistics (pixel input only) ------------------------------
    // contacts[i][j]: neighbouring pixel pairs between clusters; sharp[i][j]: of those, sharp edges
    const contacts = Array.from({ length: nC }, () => new Float64Array(nC));
    const sharp = Array.from({ length: nC }, () => new Float64Array(nC));
    const largestBlob = new Float64Array(nC).fill(Infinity);
    const pixelCount = new Float64Array(nC);
    if (pixels && pixelBins) {
        const { width: W, height: H } = pixels;
        const binCluster = new Map<number, number>();
        const binLab = new Map<number, Lab>();
        const label = new Int32Array(W * H);
        for (let i = 0; i < W * H; i++) {
            const k = pixelBins[i];
            if (k < 0) {
                label[i] = -1;
                continue;
            }
            let c = binCluster.get(k);
            if (c === undefined) {
                const lab = rgbToLab(binRgb(k));
                binLab.set(k, lab);
                c = nearestCluster(lab);
                binCluster.set(k, c);
            }
            label[i] = c;
            pixelCount[c]++;
        }
        const EDGE_DE = 12;
        const edge = (i: number, j: number) => {
            const a = label[i];
            const b = label[j];
            if (a < 0 || b < 0 || a === b) return;
            contacts[a][b]++;
            contacts[b][a]++;
            if (deltaE2000Lab(binLab.get(pixelBins![i])!, binLab.get(pixelBins![j])!) > EDGE_DE) {
                sharp[a][b]++;
                sharp[b][a]++;
            }
        };
        for (let y = 0; y < H; y++)
            for (let x = 0; x < W; x++) {
                const i = y * W + x;
                if (x + 1 < W) edge(i, i + 1);
                if (y + 1 < H) edge(i, i + W);
            }
        // largest 4-connected component per cluster
        largestBlob.fill(0);
        const seen = new Uint8Array(W * H);
        const stack: number[] = [];
        for (let s = 0; s < W * H; s++) {
            if (seen[s] || label[s] < 0) continue;
            const c = label[s];
            let size = 0;
            stack.push(s);
            seen[s] = 1;
            while (stack.length) {
                const i = stack.pop()!;
                size++;
                const x = i % W;
                const nb = [x > 0 ? i - 1 : -1, x + 1 < W ? i + 1 : -1, i - W, i + W];
                for (const j of nb)
                    if (j >= 0 && j < W * H && !seen[j] && label[j] === c) {
                        seen[j] = 1;
                        stack.push(j);
                    }
            }
            if (size > largestBlob[c]) largestBlob[c] = size;
        }
    }
    const totalPixels = pixels ? pixels.width * pixels.height : 0;

    // groups share a protected (sharp) boundary?
    const protectedPair = (g: Group, h: Group) => {
        if (!pixels) return false;
        let c = 0;
        let s = 0;
        for (const i of g.members)
            for (const j of h.members) {
                c += contacts[i][j];
                s += sharp[i][j];
            }
        return s >= 8 && s >= 0.35 * c;
    };
    // a small group is a real feature: one blob of a few pixels with a mostly sharp outline
    const isFeature = (g: Group) => {
        if (!pixels) return false;
        let c = 0;
        let s = 0;
        let blob = 0;
        for (const i of g.members) {
            blob = Math.max(blob, largestBlob[i]);
            for (let j = 0; j < nC; j++)
                if (!g.members.includes(j)) {
                    c += contacts[i][j];
                    s += sharp[i][j];
                }
        }
        return blob >= Math.max(12, totalPixels * 0.00008) && c > 0 && s >= 0.5 * c;
    };

    // ---- merge near-duplicates (complete linkage, never across sharp edges) ----
    let groups: Group[] = clusters
        .map((c, i) => ({ lab: { ...c.lab }, w: c.w, members: [i] }))
        .sort((x, y) => y.w - x.w);
    for (let changed = true; changed; ) {
        changed = false;
        outer: for (let a = 0; a < groups.length; a++)
            for (let b = a + 1; b < groups.length; b++) {
                const g = groups[a];
                const h = groups[b];
                const allClose = g.members.every((i) =>
                    h.members.every((j) => deltaE2000Lab(clusters[i].lab, clusters[j].lab) < mergeDeltaE)
                );
                if (!allClose || protectedPair(g, h)) continue;
                const w = g.w + h.w;
                g.lab = {
                    L: (g.lab.L * g.w + h.lab.L * h.w) / w,
                    a: (g.lab.a * g.w + h.lab.a * h.w) / w,
                    b: (g.lab.b * g.w + h.lab.b * h.w) / w,
                };
                g.w = w;
                g.members.push(...h.members);
                groups.splice(b, 1);
                changed = true;
                break outer;
            }
    }
    groups.sort((x, y) => y.w - x.w);

    // an edge-blend: small, thin group whose color lies between two groups it touches
    // (anti-aliasing / JPEG ringing around text and outlines)
    const isEdgeBlend = (g: Group, pool: Group[]) => {
        if (!pixels || g.w >= minWeight) return false;
        let px = 0;
        let boundary = 0;
        for (const i of g.members) {
            px += pixelCount[i];
            for (let j = 0; j < nC; j++) if (!g.members.includes(j)) boundary += contacts[i][j];
        }
        if (boundary === 0 || px / boundary > 1.5) return false;
        const touches = (h: Group) => g.members.some((i) => h.members.some((j) => contacts[i][j] > 0));
        const cands = pool.filter((h) => h !== g && touches(h));
        for (let a = 0; a < cands.length; a++)
            for (let b = a + 1; b < cands.length; b++) {
                const A = cands[a].lab;
                const B = cands[b].lab;
                const d = { L: B.L - A.L, a: B.a - A.a, b: B.b - A.b };
                const len2 = d.L ** 2 + d.a ** 2 + d.b ** 2 || 1;
                const t = ((g.lab.L - A.L) * d.L + (g.lab.a - A.a) * d.a + (g.lab.b - A.b) * d.b) / len2;
                if (t < 0.08 || t > 0.92) continue;
                const P = { L: A.L + d.L * t, a: A.a + d.a * t, b: A.b + d.b * t };
                if (deltaE2000Lab(P, g.lab) < 7) return true;
            }
        return false;
    };

    // ---- keep big groups, distinct small ones and real small features ----------
    const kept: Group[] = [];
    for (const g of groups) {
        if (isEdgeBlend(g, groups)) {
            let host = kept[0];
            for (const k of kept) if (deltaE2000Lab(k.lab, g.lab) < deltaE2000Lab(host.lab, g.lab)) host = k;
            if (host) host.w += g.w;
            continue;
        }
        if (g.w < absoluteMinWeight && !isFeature(g)) continue;
        const nearest = kept.reduce((m, k) => Math.min(m, deltaE2000Lab(k.lab, g.lab)), Infinity);
        if (g.w >= minWeight || nearest >= distinctDeltaE || isFeature(g)) kept.push(g);
        else if (kept.length) {
            let host = kept[0];
            for (const k of kept) if (deltaE2000Lab(k.lab, g.lab) < deltaE2000Lab(host.lab, g.lab)) host = k;
            host.w += g.w;
        }
    }
    // if more than maxTargets survive, keep the heaviest but always the features
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
export function expandContrast(targets: TargetColor[], strength: number, chroma = 1): TargetColor[] {
    if (strength === 0 && chroma === 1) return targets;
    const Ls = targets.map((t) => t.lab.L);
    const mean = (Math.min(...Ls) + Math.max(...Ls)) / 2;
    return targets.map((t) => {
        const L = Math.max(0, Math.min(100, mean + (t.lab.L - mean) * (1 + strength)));
        // do not tint near-white/near-black targets: boosted cream prints as yellow
        const k = t.lab.L > 85 || t.lab.L < 20 ? 1 : chroma;
        const rgb = labToRgb({ L, a: t.lab.a * k, b: t.lab.b * k }) as [number, number, number];
        return { hex: hexOf(rgb), lab: rgbToLab(rgb), weight: t.weight };
    });
}

/**
 * Share of all sharp-edge contacts that runs between each pair of targets.
 * High values mark pairs whose printed colors must stay distinguishable.
 */
export function targetEdgeShares(pixels: ImagePixels, targets: TargetColor[], edgeDeltaE = 12): number[][] {
    const { width: W, height: H } = pixels;
    const ch = pixels.channels ?? (pixels.data.length === W * H * 4 ? 4 : 3);
    const n = targets.length;
    const label = new Int32Array(W * H);
    const labs: Lab[] = new Array(W * H);
    const cache = new Map<number, number>();
    const labCache = new Map<number, Lab>();
    for (let i = 0; i < W * H; i++) {
        const o = i * ch;
        if (ch === 4 && pixels.data[o + 3] < 128) {
            label[i] = -1;
            continue;
        }
        const k = binKey(pixels.data[o], pixels.data[o + 1], pixels.data[o + 2]);
        let lab = labCache.get(k);
        if (!lab) {
            lab = rgbToLab(binRgb(k));
            labCache.set(k, lab);
        }
        labs[i] = lab;
        let t = cache.get(k);
        if (t === undefined) {
            t = 0;
            let bd = Infinity;
            for (let c = 0; c < n; c++) {
                const d = labDist2(lab, targets[c].lab);
                if (d < bd) {
                    bd = d;
                    t = c;
                }
            }
            cache.set(k, t);
        }
        label[i] = t;
    }
    const sharp = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    let total = 0;
    const edge = (i: number, j: number) => {
        const a = label[i];
        const b = label[j];
        if (a < 0 || b < 0 || a === b) return;
        if (deltaE2000Lab(labs[i], labs[j]) <= edgeDeltaE) return;
        sharp[a][b]++;
        sharp[b][a]++;
        total++;
    };
    for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++) {
            const i = y * W + x;
            if (x + 1 < W) edge(i, i + 1);
            if (y + 1 < H) edge(i, i + W);
        }
    return sharp.map((row) => row.map((v) => (total ? v / total : 0)));
}

export type AutoTunePreset =
    | 'faithful'
    | 'balanced'
    | 'bold'
    | 'popArt'
    | 'silhouette'
    | 'grayscale'
    | 'vintage';

export const AUTO_TUNE_PRESETS: AutoTunePreset[] = [
    'faithful',
    'balanced',
    'bold',
    'popArt',
    'silhouette',
    'grayscale',
    'vintage',
];

interface PresetDefinition {
    options: Partial<Omit<AutoTuneOptions, 'preset' | 'layerHeight' | 'firstLayerHeight'>>;
    /** Restyle the extracted targets before contrast/chroma search. */
    style?: (targets: TargetColor[], filaments: Filament[]) => TargetColor[];
    /** Restrict usable filaments (falls back to all if fewer than 2 remain). */
    filamentFilter?: (f: Filament) => boolean;
    /** Score fidelity against the styled targets instead of the original image colors. */
    fidelityToStyled?: boolean;
}

const labOfHex = (hex: string) => rgbToLab(parseHex(hex));
const targetFromLab = (lab: Lab, weight: number): TargetColor => {
    const rgb = labToRgb(lab) as [number, number, number];
    return { hex: hexOf(rgb), lab: rgbToLab(rgb), weight };
};

const PRESETS: Record<AutoTunePreset, PresetDefinition> = {
    // as close as possible to the original colors
    faithful: {
        options: {
            contrastStrengths: [0, 0.15, 0.3],
            chromaBoosts: [1],
            contrastWeight: 0.25,
            extract: { mergeDeltaE: 7, maxTargets: 10 },
        },
    },
    balanced: { options: {} },
    // poster look: strong contrast, saturated, few areas
    bold: {
        options: {
            contrastStrengths: [0.6, 1, 1.5],
            chromaBoosts: [1.6, 2.2],
            contrastWeight: 0.8,
            extract: { mergeDeltaE: 14, maxTargets: 6 },
        },
    },
    // Warhol style: few flat areas, hues replaced by the most vivid filaments,
    // only the light/dark order of the image is kept
    popArt: {
        options: {
            contrastStrengths: [0, 0.5],
            chromaBoosts: [1],
            contrastWeight: 0.7,
            extract: { mergeDeltaE: 22, maxTargets: 5, distinctDeltaE: 40 },
        },
        fidelityToStyled: true,
        style: (targets, filaments) => {
            const pool = filaments.map((f) => ({ f, lab: labOfHex(f.color) }));
            const used = new Set<string>();
            const byL = [...targets].sort((a, b) => a.lab.L - b.lab.L);
            const lMin = byL[0]?.lab.L ?? 0;
            const lMax = byL[byL.length - 1]?.lab.L ?? 100;
            const fl = pool.map((p) => p.lab.L);
            const fMin = Math.min(...fl);
            const fMax = Math.max(...fl);
            const mapped = new Map<TargetColor, TargetColor>();
            for (const t of byL) {
                // map image lightness range onto filament lightness range
                const rel = lMax > lMin ? (t.lab.L - lMin) / (lMax - lMin) : 0.5;
                const wantL = fMin + rel * (fMax - fMin);
                let best = pool[0];
                let bestScore = -Infinity;
                for (const p of pool) {
                    const chroma = Math.hypot(p.lab.a, p.lab.b);
                    const score = chroma - 1.5 * Math.abs(p.lab.L - wantL) - (used.has(p.f.id) ? 60 : 0);
                    if (score > bestScore) {
                        bestScore = score;
                        best = p;
                    }
                }
                used.add(best.f.id);
                mapped.set(t, targetFromLab(best.lab, t.weight));
            }
            return targets.map((t) => mapped.get(t)!);
        },
    },
    // 2–3 lightness levels, maximal legibility
    silhouette: {
        options: {
            contrastStrengths: [0.6, 1, 1.5],
            chromaBoosts: [1],
            contrastWeight: 0.9,
            extract: { mergeDeltaE: 25, maxTargets: 3, distinctDeltaE: 45 },
        },
    },
    // lightness only, neutral filaments
    grayscale: {
        options: { contrastStrengths: [0, 0.3, 0.6], chromaBoosts: [1], contrastWeight: 0.6 },
        fidelityToStyled: true,
        style: (targets) => targets.map((t) => targetFromLab({ L: t.lab.L, a: 0, b: 0 }, t.weight)),
        filamentFilter: (f) => {
            const lab = labOfHex(f.color);
            return Math.hypot(lab.a, lab.b) < 12;
        },
    },
    // softer, desaturated, warm
    vintage: {
        options: { contrastStrengths: [-0.25, -0.1], chromaBoosts: [0.6], contrastWeight: 0.4 },
        fidelityToStyled: true,
        style: (targets) =>
            targets.map((t) => targetFromLab({ L: t.lab.L, a: t.lab.a * 0.6 + 2, b: t.lab.b * 0.6 + 7 }, t.weight)),
    },
};

/**
 * Pick at most `max` filaments that are most useful for the targets
 * (always keeps the lightest one as a base for light colors).
 */
export function selectFilamentSubset(filaments: Filament[], targets: TargetColor[], max: number): Filament[] {
    if (filaments.length <= max) return filaments;
    const labs = filaments.map((f) => labOfHex(f.color));
    const score = filaments.map((_, i) =>
        targets.reduce((s, t) => s + Math.sqrt(t.weight) * Math.exp(-deltaE2000Lab(labs[i], t.lab) / 15), 0)
    );
    let lightest = 0;
    for (let i = 1; i < labs.length; i++) if (labs[i].L > labs[lightest].L) lightest = i;
    const order = filaments.map((_, i) => i).sort((a, b) => score[b] - score[a]);
    const chosen = [lightest, ...order.filter((i) => i !== lightest)].slice(0, max);
    return chosen.map((i) => filaments[i]);
}

export interface AutoTuneOptions {
    /** Style preset; explicit options below override the preset. Default 'balanced'. */
    preset?: AutoTunePreset;
    /** Use at most this many filaments (e.g. 4 for one AMS unit). */
    maxFilaments?: number;
    layerHeight: number;
    firstLayerHeight: number;
    maxHeight?: number;
    contrastStrengths?: number[];
    /** Chroma multipliers tried for the targets (pastel images print grey otherwise). */
    chromaBoosts?: number[];
    separationDeltaEs?: number[];
    /** 0 = only fidelity, 1 = strongly prefer separation between targets. */
    contrastWeight?: number;
    extract?: ExtractOptions;
    algorithm?: 'fast' | 'balanced' | 'thorough';
}

export interface AutoTuneCandidate {
    contrastStrength: number;
    chromaBoost: number;
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
    image: WeightedColor[] | ImagePixels,
    allFilaments: Filament[],
    userOptions: AutoTuneOptions
): AutoTuneResult {
    const preset = PRESETS[userOptions.preset ?? 'balanced'];
    const options: AutoTuneOptions = {
        ...preset.options,
        ...Object.fromEntries(Object.entries(userOptions).filter(([, v]) => v !== undefined)),
        extract: { ...preset.options.extract, ...userOptions.extract },
    } as AutoTuneOptions;
    const {
        layerHeight,
        firstLayerHeight,
        maxHeight,
        contrastStrengths = [0, 0.3, 0.6, 1.0, 1.5],
        chromaBoosts = [1, 1.6],
        separationDeltaEs = [6, 12],
        contrastWeight = 0.6,
        algorithm = 'balanced',
    } = options;
    const extracted = extractTargetColors(image, options.extract);
    const adjacency = Array.isArray(image) ? null : targetEdgeShares(image as ImagePixels, extracted);
    let filaments = allFilaments;
    if (preset.filamentFilter) {
        const f = allFilaments.filter(preset.filamentFilter);
        if (f.length >= 2) filaments = f;
    }
    const styledBase = preset.style ? preset.style(extracted, filaments) : extracted;
    if (options.maxFilaments && options.maxFilaments > 0)
        filaments = selectFilamentSubset(filaments, styledBase, options.maxFilaments);
    // reference for fidelity scoring
    const original = preset.fidelityToStyled ? styledBase : extracted;
    const candidates: AutoTuneCandidate[] = [];

    for (const strength of contrastStrengths)
    for (const chroma of chromaBoosts) {
        const targets = expandContrast(styledBase, strength, chroma);
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
                    // regions that meet along sharp edges (text on background) matter most
                    const edgeShare = adjacency ? adjacency[i][j] : 0;
                    const pairWeight =
                        Math.sqrt(Math.min(original[i].weight, original[j].weight)) + 3 * edgeShare;
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
                chromaBoost: chroma,
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
    const unreachable = (preset.fidelityToStyled ? [] : original)
        .map((o, i) => ({
            hex: o.hex,
            weight: o.weight,
            deltaE: deltaE2000Lab(o.lab, {
                L: best.result.finalStack.targetMappings[i].predictedLab[0],
                a: best.result.finalStack.targetMappings[i].predictedLab[1],
                b: best.result.finalStack.targetMappings[i].predictedLab[2],
            }),
        }))
        .filter((u, i) => {
            if (u.deltaE > 20) return true;
            // hue shift on clearly colored targets (e.g. purple printed as blue)
            const o = original[i].lab;
            const m = best.result.finalStack.targetMappings[i].predictedLab;
            const co = Math.hypot(o.a, o.b);
            const cp = Math.hypot(m[1], m[2]);
            if (co < 15 || cp < 10) return co >= 25 && cp < 8; // vivid color printed grey
            let dh = Math.abs(Math.atan2(o.b, o.a) - Math.atan2(m[2], m[1])) * (180 / Math.PI);
            if (dh > 180) dh = 360 - dh;
            return dh > 20;
        });
    return {
        originalTargets: extracted,
        best,
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        candidates: candidates.map(({ result, ...rest }) => rest),
        unreachable,
    };
}
