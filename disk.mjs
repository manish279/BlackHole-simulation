/**
 * AstroNova v8 — Schwarzschild + Thin Accretion Disk
 *
 * Extends v7 geodesic integrator:
 *   - Track when ray crosses equatorial plane (θ = π/2)
 *   - Determine r at crossing
 *   - If r ∈ [r_isco, r_outer], record disk hit
 *   - Compute relativistic effects at hit:
 *       1. Gravitational redshift: g_grav = √(1 − 3M/r)  (from g_tt at circular orbit)
 *       2. Doppler from disk rotation (Keplerian, Ω = M^(1/2) r^(-3/2))
 *       3. Combined redshift factor g = g_grav · g_dop
 *       4. Observed intensity I_obs = g^4 · I_emit  (Liouville/relativistic beaming)
 *
 * ISCO (Schwarzschild): r_ISCO = 6M
 * Disk outer edge: r_out = 20M (configurable)
 *
 * For each pixel we still trace backward from observer. But now we work in 3D:
 * the ray's orbital plane is set by observer-position + pixel-direction. The disk
 * is equatorial (θ = π/2). We need to detect when the ray crosses θ = π/2.
 *
 * Geometry: observer at (r_obs, θ_obs, φ=0). Each ray defines a plane through
 * the BH (since all geodesics in spherical symmetry are planar). We compute the
 * inclination i of that plane to the equator, then in the planar coordinates we
 * find where the ray hits the equator.
 */

export const M = 1.0;
export const R_HORIZON = 2.0 * M;
export const B_CRIT = 3.0 * Math.sqrt(3) * M;
export const R_ISCO = 6.0 * M;

/**
 * Trace a ray in its orbital plane from observer at (r_obs) in the plane.
 * Returns dense array of (r, phi) samples for later equator-crossing detection.
 *
 * The observer is at phi = 0 in this plane. The ray goes inward.
 */
export function traceRayDense(rObs, b, opts = {}) {
  const maxSteps = opts.maxSteps ?? 100000;
  const dphi = opts.dphi ?? 5e-4;

  let u = M / rObs;
  let rhs = 1.0 / (b * b) - u * u * (1.0 - 2.0 * u);
  if (rhs < 0) return { outcome: 'unresolved', samples: [] };
  let dudphi = Math.sqrt(rhs);
  let phi = 0;
  let turned = false;
  const samples = [{ phi: 0, r: rObs, u: u }];

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
    samples.push({ phi, r, u });

    if (r <= R_HORIZON) return { outcome: 'captured', samples };
    if (!turned && dudphi < 0) turned = true;
    if (turned && u <= M / rObs) return { outcome: 'escaped', samples };
  }
  return { outcome: 'unresolved', samples };
}

/**
 * Given orbital-plane samples, find all equator crossings.
 * The orbital plane is inclined to the equator at angle i.
 * In the plane, take phi=0 as observer azimuth. The equator intersects the
 * plane along a line that crosses the plane's own coordinate at two values
 * of phi: phi_eq and phi_eq + π.
 *
 * Specifically: if observer is at colatitude θ_obs (angle from disk normal),
 * and we set up the plane such that φ_plane=0 is observer direction, then the
 * equator-plane line is at:
 *     φ_eq = atan2(cos(θ_obs), -sin(θ_obs) * cos(α))
 * where α is the azimuthal angle of the ray relative to the BH-observer-pole
 * plane. We pass this directly as phi_eq.
 *
 * For each sample i where sign(sin(phi - phi_eq) - 0) changes, we have a
 * crossing. Linear interpolate to find r at crossing.
 */
export function findDiskCrossings(samples, phiEq, rInner, rOuter) {
  const crossings = [];
  // We want sign changes in sin(phi - phi_eq).
  // Equivalently: phi - phi_eq passes through 0 or π (mod 2π).
  // We check both phi_eq and phi_eq + π.
  for (const phiTarget of [phiEq, phiEq + Math.PI]) {
    let prev = samples[0].phi - phiTarget;
    // Normalize to [-π, π]
    const norm = (a) => Math.atan2(Math.sin(a), Math.cos(a));
    prev = norm(prev);
    for (let i = 1; i < samples.length; i++) {
      const cur = norm(samples[i].phi - phiTarget);
      // Zero crossing of normalized angle: sign change AND not the wraparound jump
      if (prev * cur < 0 && Math.abs(cur - prev) < Math.PI / 2) {
        const t = -prev / (cur - prev);
        const r = samples[i - 1].r + t * (samples[i].r - samples[i - 1].r);
        if (r >= rInner && r <= rOuter) {
          crossings.push({ r, phiSwept: samples[i - 1].phi + t * (samples[i].phi - samples[i - 1].phi), side: phiTarget === phiEq ? 'front' : 'back' });
        }
      }
      prev = cur;
    }
  }
  return crossings;
}

/**
 * Compute observed intensity at a disk crossing.
 *
 * Doppler + gravitational redshift combined factor g = E_obs / E_emit.
 *
 * Disk material moves in circular Keplerian orbit at r:
 *   Ω = M^(1/2) / r^(3/2)
 *   v_φ = Ω · r in coordinate frame
 *   Locally, the orbital velocity in the disk's static-frame is:
 *     v = √(M/r) / √(1 − 2M/r) · √(1 / (1 − 2M/r))... use the standard formula.
 *
 * Standard result for emitter on circular orbit at radius r, photon with
 * impact parameter b emitted in the equatorial plane:
 *
 *   g = (1/u^t_emit) · (1 + Ω · b_z)^(-1)
 *
 * where:
 *   u^t_emit = 1/√(1 − 3M/r)
 *   b_z = component of impact parameter along disk angular momentum
 *
 * For a ray hitting the disk at radius r with azimuthal angle relative to the
 * disk-velocity direction θ_v, the projected b along disk velocity is:
 *
 *   g = √(1 − 3M/r) / (1 + Ω · b · cos(θ_v))
 *
 * Observed flux ∝ g^4 · I_emit (relativistic beaming: I/ν^3 invariant).
 */
export function diskRedshift(r, b, cosVdotN) {
  if (r <= R_ISCO) return 0; // no stable orbits inside ISCO
  const Omega = Math.sqrt(M / (r * r * r));
  const utEmit = 1.0 / Math.sqrt(1.0 - 3.0 * M / r);
  // g factor (E_obs / E_emit)
  const g = 1.0 / (utEmit * (1.0 + Omega * b * cosVdotN));
  return g;
}

/**
 * Disk emissivity: standard alpha-disk-like power law in r (declining outward).
 * Truncated at ISCO and outer radius.
 */
export function diskEmissivity(r, rInner, rOuter) {
  if (r < rInner || r > rOuter) return 0;
  // I_emit ∝ r^(-α) with α ≈ 2-3 for thin disk; use 2 for visualization
  return Math.pow(rInner / r, 2);
}
