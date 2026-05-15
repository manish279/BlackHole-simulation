export const M = 1.0;
export const R_HORIZON = 2.0 * M;
export const B_CRIT = 3.0 * Math.sqrt(3) * M;

/**
 * Trace null geodesic from r=infinity inward to perihelion.
 * Uses orbit equation: d^2u/dphi^2 = 3u^2 - u, u = M/r
 * Initial conditions at u→0: du/dphi = 1/b.
 * Total asymptotic deflection: alpha = 2*phi_peri - pi.
 */
export function traceFromInfinity(b, opts = {}) {
  const maxSteps = opts.maxSteps ?? 500000;
  const dphi = opts.dphi ?? 1e-4;

  let u = 1e-10;
  let dudphi = 1.0 / b;
  let phi = 0;

  for (let step = 0; step < maxSteps; step++) {
    const f = (uu, vv) => ({ du: vv, dv: 3 * uu * uu - uu });
    const k1 = f(u, dudphi);
    const k2 = f(u + 0.5 * dphi * k1.du, dudphi + 0.5 * dphi * k1.dv);
    const k3 = f(u + 0.5 * dphi * k2.du, dudphi + 0.5 * dphi * k2.dv);
    const k4 = f(u + dphi * k3.du, dudphi + dphi * k3.dv);
    u += (dphi / 6) * (k1.du + 2 * k2.du + 2 * k3.du + k4.du);
    dudphi += (dphi / 6) * (k1.dv + 2 * k2.dv + 2 * k3.dv + k4.dv);
    phi += dphi;

    const r = M / u;
    if (r <= R_HORIZON) return { outcome: 'captured', phiPeri: phi, deflection: 0, b };
    if (dudphi <= 0) {
      const deflection = 2 * phi - Math.PI;
      return { outcome: 'escaped', phiPeri: phi, deflection, b };
    }
  }
  return { outcome: 'unresolved', phiPeri: phi, deflection: 0, b };
}

/** Trace ray from finite observer for rendering. */
export function traceRay(rObs, b, opts = {}) {
  const maxSteps = opts.maxSteps ?? 50000;
  const dphi = opts.dphi ?? 5e-4;

  let u = M / rObs;
  let rhs = 1.0 / (b * b) - u * u * (1.0 - 2.0 * u);
  if (rhs < 0) return { outcome: 'unresolved', phi: 0, rFinal: rObs, b };
  let dudphi = Math.sqrt(rhs);
  let phi = 0;
  let turned = false;

  for (let step = 0; step < maxSteps; step++) {
    const f = (uu, vv) => ({ du: vv, dv: 3 * uu * uu - uu });
    const k1 = f(u, dudphi);
    const k2 = f(u + 0.5 * dphi * k1.du, dudphi + 0.5 * dphi * k1.dv);
    const k3 = f(u + 0.5 * dphi * k2.du, dudphi + 0.5 * dphi * k2.dv);
    const k4 = f(u + dphi * k3.du, dudphi + dphi * k3.dv);
    u += (dphi / 6) * (k1.du + 2 * k2.du + 2 * k3.du + k4.du);
    dudphi += (dphi / 6) * (k1.dv + 2 * k2.dv + 2 * k3.dv + k4.dv);
    phi += dphi;

    const r = M / u;
    if (r <= R_HORIZON) return { outcome: 'captured', phi, rFinal: r, b };
    if (!turned && dudphi < 0) turned = true;
    if (turned && u <= M / rObs) return { outcome: 'escaped', phi, rFinal: M / u, b };
  }
  return { outcome: 'unresolved', phi, rFinal: M / u, b };
}

export function pixelAngleToImpactParameter(psi, rObs) {
  return (rObs * Math.sin(psi)) / Math.sqrt(1.0 - 2.0 * M / rObs);
}

export function shadowAngularRadius(rObs) {
  return Math.asin((B_CRIT * Math.sqrt(1.0 - 2.0 * M / rObs)) / rObs);
}
