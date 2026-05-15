import {
  traceRay,
  traceFromInfinity,
  pixelAngleToImpactParameter,
  shadowAngularRadius,
  B_CRIT,
  M
} from '../src/geodesic.mjs';

const results = [];
const check = (name, cond, detail) => results.push({ name, passed: cond, detail });

// TEST 1
{
  const rObs = 1000 * M;
  const below = traceRay(rObs, B_CRIT * 0.99, { dphi: 5e-4 });
  const above = traceRay(rObs, B_CRIT * 1.01, { dphi: 5e-4 });
  check('capture below b_crit', below.outcome === 'captured', `b=${(B_CRIT*0.99).toFixed(4)} → ${below.outcome}`);
  check('escape above b_crit', above.outcome === 'escaped', `b=${(B_CRIT*1.01).toFixed(4)} → ${above.outcome}`);
}

// TEST 2: Weak field
{
  for (const b of [50, 100, 500, 1000]) {
    const dphi = b >= 500 ? 2e-5 : 1e-4;
    const maxSteps = b >= 500 ? 5000000 : 500000;
    const ray = traceFromInfinity(b, { dphi, maxSteps });
    const analytic = 4 * M / b;
    const relErr = Math.abs((ray.deflection - analytic) / analytic);
    // Tolerance 8%: 4M/b is leading-order, higher GR corrections are ~15πM²/(4b²)
    check(`weak-field b=${b}`, ray.outcome === 'escaped' && relErr < 0.08,
      `alpha_num=${ray.deflection.toExponential(4)}, 4M/b=${analytic.toExponential(4)}, relErr=${(relErr*100).toFixed(2)}%`);
  }
}

// TEST 3: Strong-field monotonic
{
  const eps = [0.1, 0.01, 0.001];
  const defl = [];
  for (const e of eps) {
    const b = B_CRIT * (1 + e);
    const ray = traceFromInfinity(b, { dphi: 1e-4, maxSteps: 1000000 });
    defl.push(ray.deflection);
  }
  const mono = defl[2] > defl[1] && defl[1] > defl[0];
  check('strong-field log divergence', mono, `deflections eps=0.1,0.01,0.001: ${defl.map(d => d.toFixed(3)).join(', ')}`);

  // Bozza coefficient
  const s1 = (defl[1] - defl[0]) / Math.log(eps[0] / eps[1]);
  const s2 = (defl[2] - defl[1]) / Math.log(eps[1] / eps[2]);
  const ok = Math.abs(s1 - 1) < 0.2 && Math.abs(s2 - 1) < 0.2;
  check('Bozza log coefficient ≈ 1', ok, `slope1=${s1.toFixed(3)}, slope2=${s2.toFixed(3)}`);
}

// TEST 4
{
  const rObs = 100 * M;
  const psi = shadowAngularRadius(rObs);
  const bRec = pixelAngleToImpactParameter(psi, rObs);
  const relErr = Math.abs(bRec - B_CRIT) / B_CRIT;
  check('shadow ↔ b_crit roundtrip', relErr < 1e-10,
    `psi=${psi.toFixed(6)}, b_rec=${bRec.toFixed(6)}`);
}

// TEST 5
{
  const rObs = 10000 * M;
  const psi = shadowAngularRadius(rObs);
  const approx = B_CRIT / rObs;
  const relErr = Math.abs(psi - approx) / approx;
  check('far-field shadow size', relErr < 1e-3,
    `psi=${psi.toExponential(4)}, b_crit/r=${approx.toExponential(4)}`);
}

console.log('\n=== AstroNova v7 Validation Suite ===\n');
let p = 0, f = 0;
for (const r of results) {
  console.log(`${r.passed ? '✓' : '✗'} ${r.name}`);
  console.log(`    ${r.detail}`);
  r.passed ? p++ : f++;
}
console.log(`\n${p}/${p+f} tests passed.\n`);
process.exit(f > 0 ? 1 : 0);
