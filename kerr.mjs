/**
 * AstroNova v9 — Kerr Black Hole (initial cut)
 *
 * Boyer-Lindquist coordinates: (t, r, θ, φ)
 * Black hole mass M = 1, spin a ∈ [0, 1) (extremal at a = M).
 *
 * Metric functions:
 *   Δ = r² − 2Mr + a²
 *   Σ = r² + a² cos²θ
 *
 * Horizons: Δ = 0 → r± = M ± √(M² − a²)
 * Outer horizon r+ is the relevant one for capture.
 *
 * Null geodesics have three constants of motion:
 *   E  = energy
 *   L_z = axial angular momentum
 *   Q   = Carter's constant
 *
 * We use the standard impact parameters:
 *   ξ = L_z / E  (azimuthal)
 *   η = Q / E²
 *
 * Effective potentials:
 *   R(r) = [(r² + a²)E − a L_z]² − Δ[(L_z − a E)² + Q]
 *        = [r² + a² − a ξ]² − Δ[(ξ − a)² + η]      (in units of E²)
 *
 *   Θ(θ) = Q − cos²θ[a²(−E²) + L_z²/sin²θ]·(−1)... 
 *        = η + a² cos²θ − ξ² cot²θ                  (in units of E²)
 *   (rearranged: Θ(θ) = η + cos²θ(a² − ξ²/sin²θ) )
 *
 * Equations of motion (Mino time λ, dλ = dτ/Σ):
 *   (dr/dλ)² = R(r)
 *   (dθ/dλ)² = Θ(θ)
 *   dφ/dλ = a(r² + a² − a ξ)/Δ − a + ξ/sin²θ
 *
 * Critical curve (shadow boundary) in (ξ, η):
 *   R(r_ph) = 0 AND R'(r_ph) = 0  for some r_ph
 * Solving gives the parametric photon-orbit curves:
 *   ξ(r_ph) = [M(r_ph² − a²) − r_ph Δ_ph] / [a(r_ph − M)]
 *           = -(r_ph³ − 3Mr_ph² + a²r_ph + a²M) / [a(r_ph − M)]
 *   η(r_ph) =  r_ph³ [4Ma² − r_ph(r_ph − 3M)²] / [a²(r_ph − M)²]
 *
 * The shadow is the set of (ξ, η) traced by r_ph ∈ [r_ph_min, r_ph_max] where
 * those are the prograde/retrograde photon sphere radii.
 *
 * Observer at (r_obs, θ_obs):
 *   The (ξ, η) pair maps to apparent-sky coordinates (α, β):
 *     α = -ξ / sin(θ_obs)
 *     β = ±√(η + a² cos²θ_obs − ξ² cot²θ_obs)
 *   (in units of M, in the limit r_obs → ∞)
 *
 * These are the celestial-sphere coordinates of the photon's asymptotic direction.
 */

export const M = 1.0;

export class KerrBH {
  constructor(a) {
    if (a >= M) throw new Error('a must be < M (sub-extremal)');
    this.a = a;
    this.rPlus = M + Math.sqrt(M * M - a * a);
    this.rMinus = M - Math.sqrt(M * M - a * a);
  }

  Delta(r) { return r * r - 2 * M * r + this.a * this.a; }
  Sigma(r, theta) { return r * r + this.a * this.a * Math.cos(theta) ** 2; }

  /**
   * Critical curve. For a → 0, falls back to Schwarzschild result: shadow is
   * a circle of radius b_crit = 3√3 M at r_ph = 3M.
   */
  criticalCurve(rPh) {
    const a = this.a;
    if (a < 1e-5) {
      // Schwarzschild limit: photon sphere at r=3M, shadow radius b_crit
      // Parametrize by rPh ∈ [a small range around 3M] → return b_crit-circle
      // We return ξ = 0, η = b_crit² which gives α=0, β=±b_crit (only an axis point).
      // To trace full circle we instead vary ξ via a different parametrization in shadowBoundary.
      // Signal this case:
      return null;
    }
    const num_xi = -(rPh ** 3 - 3 * M * rPh ** 2 + a * a * rPh + a * a * M);
    const den_xi = a * (rPh - M);
    if (Math.abs(den_xi) < 1e-12) return null;
    const xi = num_xi / den_xi;
    const num_eta = (rPh ** 3) * (4 * M * a * a - rPh * (rPh - 3 * M) ** 2);
    const den_eta = (a * (rPh - M)) ** 2;
    const eta = num_eta / den_eta;
    if (eta < 0) return null;
    return { xi, eta };
  }

  /**
   * Map (ξ, η) on critical curve to apparent celestial coords (α, β) seen by
   * observer at colatitude θ_obs (in radians) at r_obs → ∞.
   *
   *   α = -ξ / sin(θ_obs)
   *   β² = η + a² cos²θ_obs − ξ² cot²θ_obs
   *   β = ±√(β²)
   */
  apparentCoords(xi, eta, thetaObs) {
    const sinTheta = Math.sin(thetaObs);
    const cosTheta = Math.cos(thetaObs);
    const alpha = -xi / sinTheta;
    const beta2 = eta + this.a * this.a * cosTheta * cosTheta - xi * xi * (cosTheta / sinTheta) ** 2;
    if (beta2 < 0) return null;
    const beta = Math.sqrt(beta2);
    return { alpha, beta };
  }

  /**
   * Trace shadow boundary in apparent-sky (α, β) coords.
   */
  shadowBoundary(thetaObs, nSamples = 400) {
    const points = [];

    // a → 0 limit: shadow is circle of radius b_crit
    if (this.a < 1e-5) {
      const b_crit = 3 * Math.sqrt(3) * M;
      for (let i = 0; i < nSamples; i++) {
        const t = (2 * Math.PI * i) / nSamples;
        points.push({ alpha: b_crit * Math.cos(t), beta: b_crit * Math.sin(t), rPh: 3 * M });
      }
      return points;
    }

    const rMin = this.rPlus + 1e-4;
    const rMax = 4 * M - 1e-4;
    for (let i = 0; i < nSamples; i++) {
      const t = i / (nSamples - 1);
      const rPh = rMin + t * (rMax - rMin);
      const crit = this.criticalCurve(rPh);
      if (!crit) continue;
      const ab = this.apparentCoords(crit.xi, crit.eta, thetaObs);
      if (!ab) continue;
      points.push({ alpha: ab.alpha, beta: ab.beta, rPh });
      points.push({ alpha: ab.alpha, beta: -ab.beta, rPh });
    }
    return points;
  }
}
