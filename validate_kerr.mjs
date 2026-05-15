/**
 * Kerr shadow validation:
 *   - a=0 should give a circle of radius b_crit = 3√3 M ≈ 5.196
 *   - a=0.998 viewed edge-on (θ_obs = π/2) should give the canonical "D-shape"
 *     with prograde edge at α ≈ +2M, retrograde edge at α ≈ -7M (Bardeen)
 */

import { KerrBH, M } from '../src/kerr.mjs';

const results = [];
const check = (name, cond, detail) => results.push({ name, passed: cond, detail });

// TEST 1: a → 0 limit gives circular shadow at b_crit
{
  const bh = new KerrBH(1e-6);
  const pts = bh.shadowBoundary(Math.PI / 2, 200);
  const radii = pts.map(p => Math.hypot(p.alpha, p.beta));
  const rMin = Math.min(...radii);
  const rMax = Math.max(...radii);
  const expected = 3 * Math.sqrt(3) * M;
  const circular = (rMax - rMin) / expected < 0.01;
  const correctRadius = Math.abs((rMin + rMax) / 2 - expected) / expected < 0.01;
  check('a→0: circular shadow', circular, `r ∈ [${rMin.toFixed(4)}, ${rMax.toFixed(4)}], expected ${expected.toFixed(4)}`);
  check('a→0: correct radius b_crit', correctRadius, `mean=${((rMin+rMax)/2).toFixed(4)}, b_crit=${expected.toFixed(4)}`);
}

// TEST 2: a = 0.998 (near-extremal), θ_obs = π/2: D-shape with asymmetry
// Bardeen 1973: shadow extends approximately from α ≈ -2M (prograde) to α ≈ +7M (retrograde)
// with α = -ξ/sin(θ) convention. (Sign depends on convention; key is the asymmetry.)
{
  const bh = new KerrBH(0.998);
  const pts = bh.shadowBoundary(Math.PI / 2, 400);
  const alphas = pts.map(p => p.alpha);
  const alphaMin = Math.min(...alphas);
  const alphaMax = Math.max(...alphas);
  const width = alphaMax - alphaMin;
  // Bardeen width: ~9M total for near-extremal edge-on
  const widthOK = width > 8.5 && width < 9.5;
  // Asymmetry: |alphaMin| should be much smaller than alphaMax (or vice versa)
  // Specifically, one edge is "flat" (vertical line at α ≈ ∓2M)
  const asymmetric = Math.abs(Math.abs(alphaMin) - Math.abs(alphaMax)) > 3;
  check('a=0.998 edge-on: total width ~9M', widthOK, `α ∈ [${alphaMin.toFixed(3)}, ${alphaMax.toFixed(3)}], width=${width.toFixed(3)}`);
  check('a=0.998 edge-on: asymmetric (D-shape)', asymmetric, `|αmin|=${Math.abs(alphaMin).toFixed(2)}, |αmax|=${Math.abs(alphaMax).toFixed(2)}`);
}

// TEST 3: a = 0.9 face-on (θ_obs = 0): should be circular even though Kerr.
//         When viewed along spin axis, the shadow is again circular by symmetry.
{
  const bh = new KerrBH(0.9);
  const pts = bh.shadowBoundary(1e-3, 400); // near pole
  const radii = pts.map(p => Math.hypot(p.alpha, p.beta));
  const rMin = Math.min(...radii);
  const rMax = Math.max(...radii);
  // Face-on, no asymmetry from frame dragging at this viewing angle
  const circular = (rMax - rMin) / rMax < 0.01;
  check('a=0.9 face-on: circular', circular, `r ∈ [${rMin.toFixed(4)}, ${rMax.toFixed(4)}]`);
}

// TEST 4: Horizon
{
  const bh = new KerrBH(0.9);
  const expectedRPlus = 1 + Math.sqrt(1 - 0.81);
  check('horizon r_+ formula', Math.abs(bh.rPlus - expectedRPlus) < 1e-10,
    `r_+ = ${bh.rPlus.toFixed(6)}, expected ${expectedRPlus.toFixed(6)}`);
}

console.log('\n=== Kerr Shadow Validation ===\n');
let p = 0, f = 0;
for (const r of results) {
  console.log(`${r.passed ? '✓' : '✗'} ${r.name}`);
  console.log(`    ${r.detail}`);
  r.passed ? p++ : f++;
}
console.log(`\n${p}/${p+f} tests passed.\n`);
process.exit(f > 0 ? 1 : 0);
