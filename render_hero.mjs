import {
  traceFromInfinity,
  pixelAngleToImpactParameter,
  shadowAngularRadius,
  B_CRIT,
  M
} from '../src/geodesic.mjs';
import fs from 'fs';

const WIDTH = 800;
const HEIGHT = 800;
const R_OBS = 30 * M;
const FOV = 0.5;

function sampleSky(theta, phi) {
  const gridSpacing = Math.PI / 12;
  const wrapMod = (x, m) => ((x % m) + m) % m;
  const dPhi = Math.min(wrapMod(phi, gridSpacing), gridSpacing - wrapMod(phi, gridSpacing));
  const dTheta = Math.min(wrapMod(theta, gridSpacing), gridSpacing - wrapMod(theta, gridSpacing));
  const onGrid = dPhi < 0.003 || dTheta < 0.003;

  const seed = Math.floor(theta * 400) * 100003 + Math.floor(phi * 400) * 17;
  const rand = Math.abs(Math.sin(seed * 12.9898) * 43758.5453) % 1;
  const isStar = rand > 0.9985;
  const isBrightStar = rand > 0.9998;

  if (isBrightStar) return [255, 240, 200];
  if (isStar) {
    const b = 180 + Math.floor(rand * 75);
    return [b, b, b - 20];
  }
  if (onGrid) {
    const r = phi > 0 ? 120 : 40;
    const g = theta < Math.PI / 2 ? 120 : 40;
    return [r, g, 90];
  }
  const bg = Math.floor(6 + 5 * Math.sin(theta * 2 + phi));
  return [bg, bg, bg + 3];
}

console.log('Building deflection table...');
const TABLE_N = 400;
const table = [];
const bMin = B_CRIT * 1.00001;
const bMax = pixelAngleToImpactParameter(FOV, R_OBS);
for (let i = 0; i < TABLE_N; i++) {
  const t = i / (TABLE_N - 1);
  const b = bMin * Math.pow(bMax / bMin, t);
  const ray = traceFromInfinity(b, { dphi: 5e-5, maxSteps: 2000000 });
  table.push({ b, deflection: ray.deflection, outcome: ray.outcome });
}
console.log(`  table size: ${TABLE_N}, b ∈ [${bMin.toFixed(4)}, ${bMax.toFixed(4)}]`);

function lookupDeflection(b) {
  if (b <= B_CRIT) return { outcome: 'captured', deflection: 0 };
  if (b < table[0].b) return table[0];
  if (b > table[table.length - 1].b) return { outcome: 'escaped', deflection: 4 * M / b };
  let lo = 0, hi = table.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (table[mid].b < b) lo = mid; else hi = mid;
  }
  const t = (b - table[lo].b) / (table[hi].b - table[lo].b);
  if (table[lo].outcome !== 'escaped' || table[hi].outcome !== 'escaped') {
    return { outcome: 'captured', deflection: 0 };
  }
  return {
    outcome: 'escaped',
    deflection: table[lo].deflection * (1 - t) + table[hi].deflection * t
  };
}

console.log(`Rendering ${WIDTH}x${HEIGHT}...`);
const img = new Uint8ClampedArray(WIDTH * HEIGHT * 3);
const t0 = Date.now();

for (let y = 0; y < HEIGHT; y++) {
  for (let x = 0; x < WIDTH; x++) {
    const dx = (x - WIDTH / 2) / (WIDTH / 2) * FOV;
    const dy = (y - HEIGHT / 2) / (HEIGHT / 2) * FOV;
    const psi = Math.sqrt(dx * dx + dy * dy);
    const azimuth = Math.atan2(dy, dx);

    let r, g, b;
    if (psi < 1e-6) { r = g = b = 0; }
    else {
      const bImp = pixelAngleToImpactParameter(psi, R_OBS);
      const result = lookupDeflection(bImp);
      if (result.outcome !== 'escaped') { r = g = b = 0; }
      else {
        const theta_from_bh = psi + result.deflection;
        let skyTheta = Math.PI - theta_from_bh;
        while (skyTheta < 0) skyTheta += 2 * Math.PI;
        while (skyTheta > 2 * Math.PI) skyTheta -= 2 * Math.PI;
        if (skyTheta > Math.PI) skyTheta = 2 * Math.PI - skyTheta;
        const color = sampleSky(skyTheta, azimuth);
        r = color[0]; g = color[1]; b = color[2];
      }
    }
    const idx = (y * WIDTH + x) * 3;
    img[idx] = r; img[idx+1] = g; img[idx+2] = b;
  }
  if (y % 100 === 0) console.log(`  row ${y}/${HEIGHT}`);
}

console.log(`Done in ${((Date.now()-t0)/1000).toFixed(1)}s`);
const header = `P6\n${WIDTH} ${HEIGHT}\n255\n`;
fs.writeFileSync('/home/claude/astronova-v7/hero_v7.ppm', Buffer.concat([Buffer.from(header), Buffer.from(img)]));
console.log('Wrote hero_v7.ppm');
