import { M, R_HORIZON, B_CRIT, R_ISCO,
         traceRayDense, findDiskCrossings, diskEmissivity } from './disk.mjs';
import { pixelAngleToImpactParameter } from './geodesic.mjs';
import fs from 'fs';

const WIDTH = 400;
const HEIGHT = 400;
const R_OBS = 30 * M;
const THETA_OBS = (80 * Math.PI) / 180;
const FOV = 0.5;
const R_DISK_INNER = R_ISCO;
const R_DISK_OUTER = 20 * M;

// Precompute trajectory table keyed by b
console.log('Precomputing trajectory table...');
const TABLE_N = 250;
const bMin = B_CRIT * 1.0001;
const bMaxFOV = pixelAngleToImpactParameter(FOV * 1.5, R_OBS);
const table = [];
for (let i = 0; i < TABLE_N; i++) {
  const t = i / (TABLE_N - 1);
  const b = bMin * Math.pow(bMaxFOV / bMin, t);
  const res = traceRayDense(R_OBS, b, { dphi: 8e-4, maxSteps: 40000 });
  table.push({ b, ...res });
}
console.log(`  table size: ${TABLE_N}`);

function lookupTrajectory(b) {
  if (b <= B_CRIT) return null;
  // Nearest neighbor (good enough for our purposes; b grid is dense)
  let lo = 0, hi = table.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (table[mid].b < b) lo = mid; else hi = mid;
  }
  // Pick closer
  return Math.abs(table[lo].b - b) < Math.abs(table[hi].b - b) ? table[lo] : table[hi];
}

function renderPixel(px, py) {
  const sinTheta = Math.sin(THETA_OBS), cosTheta = Math.cos(THETA_OBS);
  const O = [R_OBS * sinTheta, 0, R_OBS * cosTheta];
  const er = [sinTheta, 0, cosTheta];
  const etheta = [cosTheta, 0, -sinTheta];
  const ephi = [0, 1, 0];

  const d = [
    -er[0] + py * etheta[0] + px * ephi[0],
    -er[1] + py * etheta[1] + px * ephi[1],
    -er[2] + py * etheta[2] + px * ephi[2]
  ];
  const dMag = Math.hypot(d[0], d[1], d[2]);
  d[0] /= dMag; d[1] /= dMag; d[2] /= dMag;

  const cosPsi = -(d[0] * er[0] + d[1] * er[1] + d[2] * er[2]);
  const psi = Math.acos(Math.min(1, Math.max(-1, cosPsi)));
  if (psi < 1e-8) return [0, 0, 0];

  const bImp = pixelAngleToImpactParameter(psi, R_OBS);
  if (bImp <= B_CRIT) return [0, 0, 0]; // shadow

  const n = [
    O[1] * d[2] - O[2] * d[1],
    O[2] * d[0] - O[0] * d[2],
    O[0] * d[1] - O[1] * d[0]
  ];
  const nMag = Math.hypot(n[0], n[1], n[2]);
  if (nMag < 1e-12) return [0, 0, 0];
  n[0] /= nMag; n[1] /= nMag; n[2] /= nMag;

  const e1 = [O[0] / R_OBS, O[1] / R_OBS, O[2] / R_OBS];
  const e2 = [
    n[1] * e1[2] - n[2] * e1[1],
    n[2] * e1[0] - n[0] * e1[2],
    n[0] * e1[1] - n[1] * e1[0]
  ];
  const e2dotD = e2[0] * d[0] + e2[1] * d[1] + e2[2] * d[2];
  if (e2dotD < 0) { e2[0] = -e2[0]; e2[1] = -e2[1]; e2[2] = -e2[2]; }

  // Equator crossing direction in plane coords
  const zCrossN = [-n[1], n[0], 0];
  const zcrLen = Math.hypot(zCrossN[0], zCrossN[1], zCrossN[2]);
  if (zcrLen < 1e-12) return [0, 0, 0];
  const eq_e1 = (zCrossN[0] * e1[0] + zCrossN[1] * e1[1] + zCrossN[2] * e1[2]) / zcrLen;
  const eq_e2 = (zCrossN[0] * e2[0] + zCrossN[1] * e2[1] + zCrossN[2] * e2[2]) / zcrLen;
  const phiEq = Math.atan2(eq_e2, eq_e1);

  const traj = lookupTrajectory(bImp);
  if (!traj || traj.outcome === 'captured') return [0, 0, 0];

  const crossings = findDiskCrossings(traj.samples, phiEq, R_DISK_INNER, R_DISK_OUTER);
  if (crossings.length === 0) return [0, 0, 0];
  crossings.sort((a, b) => a.phiSwept - b.phiSwept);
  const hit = crossings[0];

  // 3D hit position
  const cosP = Math.cos(hit.phiSwept), sinP = Math.sin(hit.phiSwept);
  const hitZ = hit.r * (cosP * e1[2] + sinP * e2[2]);
  // sanity: |hitZ| should be small
  const hitXY = [
    hit.r * (cosP * e1[0] + sinP * e2[0]),
    hit.r * (cosP * e1[1] + sinP * e2[1])
  ];
  const phiDisk = Math.atan2(hitXY[1], hitXY[0]);
  const rDisk = Math.hypot(hitXY[0], hitXY[1]);
  if (rDisk < R_DISK_INNER || rDisk > R_DISK_OUTER) return [0, 0, 0];

  // b_z (impact parameter projected on disk angular momentum +ẑ)
  const bZ = bImp * n[2];
  const Omega = Math.sqrt(M / (rDisk * rDisk * rDisk));
  // Cunningham g-factor for prograde Keplerian disk
  let g = Math.sqrt(1.0 - 3.0 * M / rDisk) / (1.0 + Omega * bZ);
  if (!isFinite(g) || g <= 0) return [0, 0, 0];

  const Iemit = diskEmissivity(rDisk, R_DISK_INNER, R_DISK_OUTER);
  const Iobs = Math.pow(g, 4) * Iemit;

  // Smooth blackbody-ish color from observed effective temperature
  const localT = Math.pow(R_DISK_INNER / rDisk, 0.75);
  const obsT = localT * g;
  // Smooth interpolation through color stops:
  // [0.3] red(1.0, 0.2, 0.1) [0.5] orange(1.0, 0.55, 0.2) [0.75] yellow(1.0, 0.9, 0.55) [1.0] white(0.95, 0.95, 0.95) [1.3] blue-white(0.7, 0.85, 1.05)
  function lerp(a, b, t) { return a + (b - a) * t; }
  function colorRamp(T) {
    const stops = [
      { t: 0.25, c: [0.7, 0.1, 0.05] },
      { t: 0.45, c: [1.0, 0.35, 0.15] },
      { t: 0.65, c: [1.0, 0.7, 0.35] },
      { t: 0.85, c: [1.0, 0.95, 0.75] },
      { t: 1.05, c: [0.95, 0.95, 0.95] },
      { t: 1.3,  c: [0.7, 0.85, 1.1] }
    ];
    if (T <= stops[0].t) return stops[0].c;
    if (T >= stops[stops.length-1].t) return stops[stops.length-1].c;
    for (let i = 0; i < stops.length - 1; i++) {
      if (T >= stops[i].t && T <= stops[i+1].t) {
        const u = (T - stops[i].t) / (stops[i+1].t - stops[i].t);
        return [lerp(stops[i].c[0], stops[i+1].c[0], u),
                lerp(stops[i].c[1], stops[i+1].c[1], u),
                lerp(stops[i].c[2], stops[i+1].c[2], u)];
      }
    }
    return [1,1,1];
  }
  const [cr, cg, cb] = colorRamp(obsT);

  const I = Math.min(1.0, Iobs * 10.0);
  return [
    Math.min(255, Math.floor(cr * I * 255)),
    Math.min(255, Math.floor(cg * I * 255)),
    Math.min(255, Math.floor(cb * I * 255))
  ];
}

console.log(`Rendering ${WIDTH}x${HEIGHT}...`);
const img = new Uint8ClampedArray(WIDTH * HEIGHT * 3);
const t0 = Date.now();
let nDisk = 0;
for (let y = 0; y < HEIGHT; y++) {
  for (let x = 0; x < WIDTH; x++) {
    const px = ((x - WIDTH / 2) / (WIDTH / 2)) * FOV;
    const py = -((y - HEIGHT / 2) / (HEIGHT / 2)) * FOV;
    const col = renderPixel(px, py);
    const idx = (y * WIDTH + x) * 3;
    img[idx] = col[0]; img[idx + 1] = col[1]; img[idx + 2] = col[2];
    if (col[0] + col[1] + col[2] > 0) nDisk++;
  }
  if (y % 50 === 0) console.log(`  row ${y}/${HEIGHT}, ${((Date.now()-t0)/1000).toFixed(1)}s`);
}

console.log(`Done in ${((Date.now()-t0)/1000).toFixed(1)}s. Disk pixels: ${nDisk}`);
const header = `P6\n${WIDTH} ${HEIGHT}\n255\n`;
fs.writeFileSync('/home/claude/astronova-v7/disk_v8.ppm', Buffer.concat([Buffer.from(header), Buffer.from(img)]));
console.log('Wrote disk_v8.ppm');
