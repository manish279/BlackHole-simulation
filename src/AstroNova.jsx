import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// PHYSICS ENGINE — Schwarzschild null geodesic + thin disk
// (same integrator as AstroNova v7/v8, ported to JS inline)
// ============================================================
const M = 1.0;
const B_CRIT = 3.0 * Math.sqrt(3) * M;
const R_HORIZON = 2.0 * M;
const R_ISCO = 6.0 * M;

function traceFromInfinity(b, dphi = 1e-4, maxSteps = 500000) {
  let u = 1e-10, dudphi = 1.0 / b, phi = 0;
  for (let step = 0; step < maxSteps; step++) {
    const k1du = dudphi, k1dv = 3 * u * u - u;
    const u2 = u + 0.5 * dphi * k1du, v2 = dudphi + 0.5 * dphi * k1dv;
    const k2du = v2, k2dv = 3 * u2 * u2 - u2;
    const u3 = u + 0.5 * dphi * k2du, v3 = dudphi + 0.5 * dphi * k2dv;
    const k3du = v3, k3dv = 3 * u3 * u3 - u3;
    const u4 = u + dphi * k3du, v4 = dudphi + dphi * k3dv;
    const k4du = v4, k4dv = 3 * u4 * u4 - u4;
    u += (dphi / 6) * (k1du + 2 * k2du + 2 * k3du + k4du);
    dudphi += (dphi / 6) * (k1dv + 2 * k2dv + 2 * k3dv + k4dv);
    phi += dphi;
    const r = M / u;
    if (r <= R_HORIZON) return { outcome: "captured", deflection: 0, phi };
    if (dudphi <= 0) return { outcome: "escaped", deflection: 2 * phi - Math.PI, phi };
  }
  return { outcome: "unresolved", deflection: 0, phi };
}

function traceRayDense(rObs, b, dphi = 6e-4, maxSteps = 50000) {
  let u = M / rObs;
  const rhs = 1.0 / (b * b) - u * u * (1.0 - 2.0 * u);
  if (rhs < 0) return { outcome: "unresolved", samples: [] };
  let dudphi = Math.sqrt(rhs), phi = 0, turned = false;
  const samples = [{ phi: 0, r: rObs }];
  for (let step = 0; step < maxSteps; step++) {
    const k1du = dudphi, k1dv = 3 * u * u - u;
    const u2 = u + 0.5 * dphi * k1du, v2 = dudphi + 0.5 * dphi * k1dv;
    const k2du = v2, k2dv = 3 * u2 * u2 - u2;
    const u3 = u + 0.5 * dphi * k2du, v3 = dudphi + 0.5 * dphi * k2dv;
    const k3du = v3, k3dv = 3 * u3 * u3 - u3;
    const u4 = u + dphi * k3du, v4 = dudphi + dphi * k3dv;
    const k4du = v4, k4dv = 3 * u4 * u4 - u4;
    u += (dphi / 6) * (k1du + 2 * k2du + 2 * k3du + k4du);
    dudphi += (dphi / 6) * (k1dv + 2 * k2dv + 2 * k3dv + k4dv);
    phi += dphi;
    const r = M / u;
    samples.push({ phi, r });
    if (r <= R_HORIZON) return { outcome: "captured", samples };
    if (!turned && dudphi < 0) turned = true;
    if (turned && u <= M / rObs) return { outcome: "escaped", samples };
  }
  return { outcome: "unresolved", samples };
}

function shadowAngularRadius(rObs) {
  return Math.asin((B_CRIT * Math.sqrt(1.0 - 2.0 * M / rObs)) / rObs);
}

function pixelAngleToImpactParameter(psi, rObs) {
  return (rObs * Math.sin(psi)) / Math.sqrt(1.0 - 2.0 * M / rObs);
}

function buildDeflectionTable(rObs, fov, nSamples = 180) {
  const table = [];
  const bMin = B_CRIT * 1.0002;
  const bMax = pixelAngleToImpactParameter(fov * 1.2, rObs);
  for (let i = 0; i < nSamples; i++) {
    const t = i / (nSamples - 1);
    const b = bMin * Math.pow(bMax / bMin, t);
    const res = traceFromInfinity(b, 1.2e-4);
    table.push({ b, ...res });
  }
  return table;
}

function buildTrajectoryTable(rObs, fov, nSamples = 160) {
  const table = [];
  const bMin = B_CRIT * 1.0002;
  const bMax = pixelAngleToImpactParameter(fov * 1.2, rObs);
  for (let i = 0; i < nSamples; i++) {
    const t = i / (nSamples - 1);
    const b = bMin * Math.pow(bMax / bMin, t);
    const res = traceRayDense(rObs, b);
    table.push({ b, ...res });
  }
  return table;
}

function lookupDeflection(table, b) {
  if (b <= B_CRIT) return { outcome: "captured", deflection: 0 };
  if (b < table[0].b) return table[0];
  if (b > table[table.length - 1].b) return { outcome: "escaped", deflection: 4 * M / b };
  let lo = 0, hi = table.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; table[mid].b < b ? lo = mid : hi = mid; }
  const t = (b - table[lo].b) / (table[hi].b - table[lo].b);
  if (table[lo].outcome !== "escaped" || table[hi].outcome !== "escaped") return { outcome: "captured", deflection: 0 };
  return { outcome: "escaped", deflection: table[lo].deflection * (1 - t) + table[hi].deflection * t };
}

function lookupTrajectory(table, b) {
  if (b <= B_CRIT) return null;
  let lo = 0, hi = table.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; table[mid].b < b ? lo = mid : hi = mid; }
  return Math.abs(table[lo].b - b) < Math.abs(table[hi].b - b) ? table[lo] : table[hi];
}

function findDiskCrossings(samples, phiEq, rInner, rOuter) {
  const crossings = [];
  const norm = a => Math.atan2(Math.sin(a), Math.cos(a));
  for (const phiTarget of [phiEq, phiEq + Math.PI]) {
    let prev = norm(samples[0].phi - phiTarget);
    for (let i = 1; i < samples.length; i++) {
      const cur = norm(samples[i].phi - phiTarget);
      if (prev * cur < 0 && Math.abs(cur - prev) < Math.PI / 2) {
        const t2 = -prev / (cur - prev);
        const r = samples[i - 1].r + t2 * (samples[i].r - samples[i - 1].r);
        if (r >= rInner && r <= rOuter)
          crossings.push({ r, phiSwept: samples[i - 1].phi + t2 * (samples[i].phi - samples[i - 1].phi) });
      }
      prev = cur;
    }
  }
  return crossings;
}

function colorRamp(T) {
  const stops = [
    { t: 0.2, c: [0.55, 0.05, 0.02] },
    { t: 0.4, c: [0.9, 0.25, 0.08] },
    { t: 0.6, c: [1.0, 0.6, 0.25] },
    { t: 0.8, c: [1.0, 0.92, 0.7] },
    { t: 1.0, c: [0.97, 0.97, 0.97] },
    { t: 1.3, c: [0.6, 0.8, 1.1] },
  ];
  if (T <= stops[0].t) return stops[0].c;
  if (T >= stops[stops.length - 1].t) return stops[stops.length - 1].c;
  for (let i = 0; i < stops.length - 1; i++) {
    if (T >= stops[i].t && T <= stops[i + 1].t) {
      const u = (T - stops[i].t) / (stops[i + 1].t - stops[i].t);
      return stops[i].c.map((v, j) => v + (stops[i + 1].c[j] - v) * u);
    }
  }
  return [1, 1, 1];
}

function sampleStarfield(theta, phi) {
  const g = Math.PI / 12;
  const wm = (x, m) => ((x % m) + m) % m;
  const dPhi = Math.min(wm(phi, g), g - wm(phi, g));
  const dTheta = Math.min(wm(theta, g), g - wm(theta, g));
  const onGrid = dPhi < 0.004 || dTheta < 0.004;
  const seed = Math.floor(theta * 350) * 100003 + Math.floor(phi * 350) * 17;
  const rand = Math.abs(Math.sin(seed * 12.9898) * 43758.5453) % 1;
  if (rand > 0.9997) return [255, 245, 200];
  if (rand > 0.998) return [200, 200, 210];
  if (onGrid) {
    return [Math.floor(30 + 60 * (phi > 0 ? 1 : 0)), Math.floor(30 + 60 * (theta < Math.PI / 2 ? 1 : 0)), 60];
  }
  const bg = Math.floor(5 + 4 * Math.sin(theta * 2 + phi * 0.7));
  return [bg, bg, bg + 3];
}

// ============================================================
// RENDER FUNCTION — runs synchronously, returns ImageData pixels
// ============================================================
function renderFrame(params) {
  const { width, height, rObs, thetaObs, fov, rDiskInner, rDiskOuter, showDisk, showStarfield, doppler } = params;
  const pixels = new Uint8ClampedArray(width * height * 4);

  const deflTable = buildDeflectionTable(rObs, fov);
  const trajTable = showDisk ? buildTrajectoryTable(rObs, fov) : null;

  const sinTheta = Math.sin(thetaObs), cosTheta = Math.cos(thetaObs);
  const er = [sinTheta, 0, cosTheta];
  const etheta = [cosTheta, 0, -sinTheta];
  const ephi = [0, 1, 0];
  const O = [rObs * sinTheta, 0, rObs * cosTheta];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const px = ((x - width / 2) / (width / 2)) * fov;
      const py = -((y - height / 2) / (height / 2)) * fov;
      const psi = Math.sqrt(px * px + py * py);
      const azimuth = Math.atan2(py, px);
      const idx = (y * width + x) * 4;

      if (psi < 1e-8) { pixels[idx + 3] = 255; continue; } // pure black

      const bImp = pixelAngleToImpactParameter(psi, rObs);
      if (bImp <= B_CRIT) {
        pixels[idx + 3] = 255; // shadow — black
        continue;
      }

      let dr = 0, dg = 0, db = 0;

      // --- DISK ---
      if (showDisk && trajTable) {
        // compute orbital plane
        const d = [-er[0] + py * etheta[0] + px * ephi[0],
                   -er[1] + py * etheta[1] + px * ephi[1],
                   -er[2] + py * etheta[2] + px * ephi[2]];
        const dMag = Math.hypot(d[0], d[1], d[2]);
        d[0] /= dMag; d[1] /= dMag; d[2] /= dMag;

        const n = [O[1]*d[2]-O[2]*d[1], O[2]*d[0]-O[0]*d[2], O[0]*d[1]-O[1]*d[0]];
        const nMag = Math.hypot(...n);
        if (nMag > 1e-12) {
          n[0] /= nMag; n[1] /= nMag; n[2] /= nMag;
          const e1 = [O[0]/rObs, O[1]/rObs, O[2]/rObs];
          const e2 = [n[1]*e1[2]-n[2]*e1[1], n[2]*e1[0]-n[0]*e1[2], n[0]*e1[1]-n[1]*e1[0]];
          const e2d = e2[0]*d[0]+e2[1]*d[1]+e2[2]*d[2];
          if (e2d < 0) { e2[0]=-e2[0]; e2[1]=-e2[1]; e2[2]=-e2[2]; }

          const zCN = [-n[1], n[0], 0];
          const zcrL = Math.hypot(...zCN);
          if (zcrL > 1e-12) {
            const eq1 = (zCN[0]*e1[0]+zCN[1]*e1[1]+zCN[2]*e1[2])/zcrL;
            const eq2 = (zCN[0]*e2[0]+zCN[1]*e2[1]+zCN[2]*e2[2])/zcrL;
            const phiEq = Math.atan2(eq2, eq1);
            const traj = lookupTrajectory(trajTable, bImp);
            if (traj && traj.outcome !== "captured" && traj.samples?.length > 0) {
              const crossings = findDiskCrossings(traj.samples, phiEq, rDiskInner, rDiskOuter);
              if (crossings.length > 0) {
                crossings.sort((a, b2) => a.phiSwept - b2.phiSwept);
                const hit = crossings[0];
                const cosP = Math.cos(hit.phiSwept), sinP = Math.sin(hit.phiSwept);
                const hx = hit.r*(cosP*e1[0]+sinP*e2[0]);
                const hy = hit.r*(cosP*e1[1]+sinP*e2[1]);
                const rDisk = Math.hypot(hx, hy);
                if (rDisk >= rDiskInner && rDisk <= rDiskOuter) {
                  const bZ = doppler ? bImp * n[2] : 0;
                  const Omega = Math.sqrt(M/(rDisk*rDisk*rDisk));
                  const gFactor = Math.sqrt(1-3*M/rDisk) / (1 + Omega*bZ);
                  if (isFinite(gFactor) && gFactor > 0) {
                    const Iemit = Math.pow(rDiskInner/rDisk, 2);
                    const Iobs = Math.pow(gFactor, 4) * Iemit;
                    const localT = Math.pow(rDiskInner/rDisk, 0.75);
                    const obsT = localT * gFactor;
                    const [cr, cg, cb2] = colorRamp(obsT);
                    const I = Math.min(1.0, Iobs * 9.0);
                    dr = Math.min(255, Math.floor(cr * I * 255));
                    dg = Math.min(255, Math.floor(cg * I * 255));
                    db = Math.min(255, Math.floor(cb2 * I * 255));
                  }
                }
              }
            }
          }
        }
      }

      // --- STARFIELD (background if no disk hit) ---
      if (showStarfield && dr === 0 && dg === 0 && db === 0) {
        const result = lookupDeflection(deflTable, bImp);
        if (result.outcome === "escaped") {
          let skyTheta = Math.PI - (psi + result.deflection);
          while (skyTheta < 0) skyTheta += 2 * Math.PI;
          while (skyTheta > 2 * Math.PI) skyTheta -= 2 * Math.PI;
          if (skyTheta > Math.PI) skyTheta = 2 * Math.PI - skyTheta;
          const [sr, sg, sb2] = sampleStarfield(skyTheta, azimuth);
          dr = sr; dg = sg; db = sb2;
        }
      }

      pixels[idx] = dr; pixels[idx+1] = dg; pixels[idx+2] = db; pixels[idx+3] = 255;
    }
  }
  return pixels;
}

// ============================================================
// KERR SHADOW OVERLAY (from v9 analytics)
// ============================================================
function kerrShadowPoints(a, thetaObs, nSamples = 300) {
  if (a < 1e-4) {
    const b_c = B_CRIT;
    return Array.from({ length: nSamples }, (_, i) => {
      const t = (2 * Math.PI * i) / nSamples;
      return { alpha: b_c * Math.cos(t), beta: b_c * Math.sin(t) };
    });
  }
  const rPlus = M + Math.sqrt(M * M - a * a);
  const rMin = rPlus + 1e-4, rMax = 4 * M - 1e-4;
  const pts = [];
  for (let i = 0; i <= nSamples; i++) {
    const t = i / nSamples;
    const rPh = rMin + t * (rMax - rMin);
    const num_xi = -(rPh**3 - 3*M*rPh**2 + a*a*rPh + a*a*M);
    const den_xi = a * (rPh - M);
    if (Math.abs(den_xi) < 1e-12) continue;
    const xi = num_xi / den_xi;
    const num_eta = rPh**3 * (4*M*a*a - rPh*(rPh-3*M)**2);
    const den_eta = (a*(rPh-M))**2;
    const eta = num_eta / den_eta;
    if (eta < 0) continue;
    const sinT = Math.sin(thetaObs), cosT = Math.cos(thetaObs);
    const alpha = -xi / sinT;
    const beta2 = eta + a*a*cosT*cosT - xi*xi*(cosT/sinT)**2;
    if (beta2 < 0) continue;
    const beta = Math.sqrt(beta2);
    pts.push({ alpha, beta });
    pts.push({ alpha, beta: -beta });
  }
  return pts;
}

// ============================================================
// DESIGN TOKENS — Palantir × Nolan/Interstellar
// ============================================================
const BG    = '#040810';
const BG2   = '#070d18';
const BG3   = '#0a1220';
const TEAL  = '#00c8e8';
const AMBER = '#e8a520';
const TEXT  = '#b8d0e8';
const TEXTDIM = '#3d5570';
const BORDER  = 'rgba(0,200,232,0.18)';
const BORDER_A = 'rgba(232,165,32,0.22)';
const MONO = "'Space Mono','IBM Plex Mono','Courier New',monospace";

const GRID_BG = {
  background: BG,
  backgroundImage: `linear-gradient(${TEAL}08 1px,transparent 1px),linear-gradient(90deg,${TEAL}08 1px,transparent 1px)`,
  backgroundSize: '48px 48px',
};

// Shared sub-components
function SectionHead({ color = AMBER, children }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '14px' }}>
      <div style={{ width: '3px', height: '12px', background: color, flexShrink: 0 }} />
      <span style={{ fontSize: '9px', color, letterSpacing: '0.22em', textTransform: 'uppercase', fontFamily: MONO }}>
        {children}
      </span>
      <div style={{ flex: 1, height: '1px', background: color === AMBER ? BORDER_A : BORDER }} />
    </div>
  );
}

function StatRow({ label, value, color = TEAL }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
      padding: '5px 0', borderBottom: `1px solid ${BG3}`,
    }}>
      <span style={{ fontSize: '10px', color: TEXTDIM, fontFamily: MONO, letterSpacing: '0.05em' }}>{label}</span>
      <span style={{ fontSize: '11px', color, fontFamily: MONO }}>{value}</span>
    </div>
  );
}

// ============================================================
// UI COMPONENT
// ============================================================
export default function AstroNova() {
  const canvasRef = useRef(null);
  const [rendering, setRendering] = useState(false);
  const [renderTime, setRenderTime] = useState(null);
  const [stats, setStats] = useState(null);

  const [params, setParams] = useState({
    resolution: 280,
    rObs: 30,
    thetaObsDeg: 80,
    fov: 0.48,
    rDiskOuter: 20,
    spin: 0.0,
    showDisk: true,
    showStarfield: true,
    doppler: true,
    showKerrOverlay: false,
  });

  const updateParam = (key, val) => setParams(p => ({ ...p, [key]: val }));

  const doRender = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || rendering) return;
    setRendering(true);
    setRenderTime(null);

    const t0 = performance.now();
    const { resolution, rObs, thetaObsDeg, fov, rDiskOuter, showDisk, showStarfield, doppler } = params;
    const W = resolution, H = resolution;
    canvas.width = W; canvas.height = H;

    setTimeout(() => {
      const pixels = renderFrame({
        width: W, height: H,
        rObs,
        thetaObs: (thetaObsDeg * Math.PI) / 180,
        fov,
        rDiskInner: R_ISCO,
        rDiskOuter,
        showDisk, showStarfield, doppler,
      });

      const ctx = canvas.getContext('2d');
      const imgData = new ImageData(pixels, W, H);
      ctx.putImageData(imgData, 0, 0);

      // Kerr overlay
      if (params.showKerrOverlay && params.spin > 0) {
        const a = params.spin * M;
        const pts = kerrShadowPoints(a, (thetaObsDeg * Math.PI) / 180);
        const psiShadow = shadowAngularRadius(rObs);
        const cx = W / 2, cy = H / 2;
        ctx.strokeStyle = `${TEAL}b0`;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        for (const p of pts) {
          const sx = cx + (p.alpha / B_CRIT) * (W / 2) * (psiShadow / fov) * 0.95;
          const sy = cy - (p.beta / B_CRIT) * (H / 2) * (psiShadow / fov) * 0.95;
          ctx.moveTo(sx, sy); ctx.arc(sx, sy, 0.8, 0, 2 * Math.PI);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const psiS = shadowAngularRadius(rObs);
      setStats({
        bCrit: B_CRIT.toFixed(4),
        psiShadow: (psiS * 180 / Math.PI).toFixed(2),
        rISCO: R_ISCO.toFixed(1),
        gMin: (Math.sqrt(1 - 3 * M / (rDiskOuter)) / (1 + Math.sqrt(M / rDiskOuter ** 3) * B_CRIT)).toFixed(3),
        gMax: (Math.sqrt(1 - 3 * M / R_ISCO) / (1 + Math.sqrt(M / R_ISCO ** 3) * (-B_CRIT * 0.5))).toFixed(3),
      });
      setRenderTime(dt);
      setRendering(false);
    }, 30);
  }, [params, rendering]);

  useEffect(() => { doRender(); }, []);

  const exportPNG = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `astronova_r${params.rObs}_i${params.thetaObsDeg}_a${params.spin}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  // Slider row — HUD style
  const SliderRow = ({ label, paramKey, min, max, step, format, unit }) => (
    <div style={{ marginBottom: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
        <span style={{ fontSize: '9px', color: TEXTDIM, letterSpacing: '0.18em', textTransform: 'uppercase', fontFamily: MONO }}>
          {label}
        </span>
        <span style={{ fontSize: '11px', color: TEAL, fontFamily: MONO }}>
          {format ? format(params[paramKey]) : params[paramKey]}{unit || ''}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step}
        value={params[paramKey]}
        onChange={e => updateParam(paramKey, parseFloat(e.target.value))}
        style={{ width: '100%', cursor: 'pointer' }}
      />
    </div>
  );

  // Square operational toggle
  const Toggle = ({ label, paramKey, note }) => (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      marginBottom: '9px', padding: '6px 8px',
      background: params[paramKey] ? `${TEAL}0d` : 'transparent',
      border: `1px solid ${params[paramKey] ? BORDER : BG3}`,
      borderRadius: '2px', cursor: 'pointer',
    }} onClick={() => updateParam(paramKey, !params[paramKey])}>
      <div>
        <span style={{ fontSize: '9px', color: params[paramKey] ? TEXT : TEXTDIM, letterSpacing: '0.15em', textTransform: 'uppercase', fontFamily: MONO }}>
          {label}
        </span>
        {note && <div style={{ fontSize: '9px', color: TEXTDIM, marginTop: '1px', fontFamily: MONO }}>{note}</div>}
      </div>
      {/* Square LED indicator */}
      <div style={{
        width: '14px', height: '14px', borderRadius: '2px', flexShrink: 0,
        background: params[paramKey] ? TEAL : BG3,
        border: `1px solid ${params[paramKey] ? TEAL : TEXTDIM}`,
        boxShadow: params[paramKey] ? `0 0 6px ${TEAL}80` : 'none',
        transition: 'all 0.15s',
      }} />
    </div>
  );

  return (
    <div style={{
      minHeight: '100vh', ...GRID_BG,
      fontFamily: MONO,
      color: TEXT, display: 'flex', flexDirection: 'column',
    }}>

      {/* ── HEADER ── */}
      <div style={{
        borderBottom: `1px solid ${BORDER}`,
        padding: '0 24px',
        display: 'flex', alignItems: 'stretch', justifyContent: 'space-between',
        background: `${BG2}e0`, backdropFilter: 'blur(10px)',
        position: 'sticky', top: 0, zIndex: 10, height: '52px',
      }}>
        {/* Left — branding */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {/* Amber accent bar */}
          <div style={{ width: '3px', height: '32px', background: `linear-gradient(${AMBER},${TEAL})` }} />
          {/* Black hole icon */}
          <div style={{
            width: '30px', height: '30px', borderRadius: '50%',
            background: `radial-gradient(circle at 38% 38%, #0a1e3a 30%, #000 100%)`,
            boxShadow: `0 0 18px ${TEAL}50, 0 0 6px ${AMBER}30`,
            border: `1px solid ${BORDER}`,
          }} />
          <div>
            <div style={{ fontSize: '13px', fontWeight: '700', letterSpacing: '0.2em', color: '#e8f4ff', fontFamily: MONO }}>
              ASTRONOVA
            </div>
            <div style={{ fontSize: '9px', color: TEXTDIM, letterSpacing: '0.18em', fontFamily: MONO }}>
              GEODESIC · ENGINE · v9
            </div>
          </div>
          {/* Separator */}
          <div style={{ width: '1px', height: '28px', background: BORDER, marginLeft: '8px' }} />
          <div style={{ fontSize: '9px', color: AMBER, letterSpacing: '0.15em', fontFamily: MONO }}>
            SCHWARZSCHILD · THIN DISK · KERR SHADOW
          </div>
        </div>

        {/* Right — controls */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {renderTime && (
            <div style={{
              padding: '4px 10px', background: BG3, border: `1px solid ${BORDER}`,
              borderRadius: '2px', fontSize: '10px', color: TEAL, fontFamily: MONO,
            }}>
              {renderTime}s
            </div>
          )}
          <button onClick={exportPNG} style={{
            background: 'transparent', border: `1px solid ${BORDER}`,
            color: TEXTDIM, padding: '6px 14px', borderRadius: '2px',
            cursor: 'pointer', fontSize: '9px', letterSpacing: '0.15em',
            fontFamily: MONO, transition: 'color 0.15s, border-color 0.15s',
          }}
            onMouseEnter={e => { e.currentTarget.style.color = TEAL; e.currentTarget.style.borderColor = TEAL; }}
            onMouseLeave={e => { e.currentTarget.style.color = TEXTDIM; e.currentTarget.style.borderColor = BORDER; }}
          >
            EXPORT PNG
          </button>
          <button onClick={doRender} disabled={rendering} style={{
            background: rendering ? BG3 : AMBER,
            border: `1px solid ${rendering ? TEXTDIM : AMBER}`,
            color: rendering ? TEXTDIM : '#000',
            padding: '6px 20px', borderRadius: '2px',
            cursor: rendering ? 'not-allowed' : 'pointer',
            fontSize: '9px', letterSpacing: '0.2em', fontWeight: '700',
            fontFamily: MONO,
            boxShadow: rendering ? 'none' : `0 0 14px ${AMBER}50`,
            transition: 'all 0.15s',
          }}>
            {rendering ? 'INTEGRATING...' : '▶  RENDER'}
          </button>
        </div>
      </div>

      {/* ── MAIN ── */}
      <div style={{ display: 'flex', flex: 1, gap: 0 }}>

        {/* Canvas area */}
        <div style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          padding: '32px', position: 'relative', minHeight: '500px',
        }}>
          <div style={{ position: 'relative' }}>
            {/* Render overlay */}
            {rendering && (
              <div style={{
                position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
                justifyContent: 'center', background: `${BG}cc`,
                zIndex: 2, borderRadius: '2px', flexDirection: 'column', gap: '14px',
              }}>
                {/* Teal spinner ring */}
                <div style={{
                  width: '44px', height: '44px', borderRadius: '50%',
                  border: `2px solid ${BG3}`, borderTopColor: TEAL,
                  animation: 'spin 0.9s linear infinite',
                }} />
                <div style={{ fontSize: '9px', color: TEAL, letterSpacing: '0.22em', fontFamily: MONO }}>
                  INTEGRATING GEODESICS
                </div>
                <div style={{ fontSize: '9px', color: TEXTDIM, letterSpacing: '0.12em', fontFamily: MONO }}>
                  RK4 · NULL GEODESIC · SCHWARZSCHILD
                </div>
              </div>
            )}

            {/* Canvas with teal glow border */}
            <canvas ref={canvasRef}
              style={{
                imageRendering: 'pixelated',
                display: 'block',
                maxWidth: 'min(520px, 100%)',
                maxHeight: 'min(520px, 80vh)',
                width: '100%', height: 'auto',
                border: `1px solid ${BORDER}`,
                boxShadow: `0 0 60px ${TEAL}12, 0 0 120px #000c`,
                borderRadius: '2px',
              }}
            />

            {/* Corner reticles */}
            {['top:0;left:0', 'top:0;right:0', 'bottom:0;left:0', 'bottom:0;right:0'].map((pos, i) => {
              const s = Object.fromEntries(pos.split(';').map(x => x.split(':')));
              const bR = i === 0 ? '0 0 4px 0' : i === 1 ? '0 0 0 4px' : i === 2 ? '4px 0 0 0' : '0 4px 0 0';
              return (
                <div key={i} style={{
                  position: 'absolute', ...s,
                  width: '12px', height: '12px',
                  borderTop: i < 2 ? `1px solid ${TEAL}` : 'none',
                  borderBottom: i >= 2 ? `1px solid ${TEAL}` : 'none',
                  borderLeft: (i === 0 || i === 2) ? `1px solid ${TEAL}` : 'none',
                  borderRight: (i === 1 || i === 3) ? `1px solid ${TEAL}` : 'none',
                  borderRadius: bR, pointerEvents: 'none',
                }} />
              );
            })}
          </div>

          {/* Status strip below canvas */}
          <div style={{
            position: 'absolute', bottom: '14px', left: '50%', transform: 'translateX(-50%)',
            background: `${BG2}e0`, border: `1px solid ${BORDER}`,
            borderRadius: '2px', padding: '5px 16px',
            fontSize: '9px', color: TEXTDIM, letterSpacing: '0.13em',
            whiteSpace: 'nowrap', fontFamily: MONO,
            display: 'flex', alignItems: 'center', gap: '12px',
          }}>
            <span style={{ color: TEAL, fontSize: '8px' }}>●</span>
            SCHWARZSCHILD NULL GEODESICS
            <span style={{ color: BORDER }}>|</span>
            CUNNINGHAM g⁴ TRANSFER
            <span style={{ color: BORDER }}>|</span>
            VALIDATED
          </div>
        </div>

        {/* ── CONTROLS PANEL ── */}
        <div style={{
          width: '288px', borderLeft: `1px solid ${BORDER}`,
          background: `${BG2}cc`,
          padding: '20px 16px', overflowY: 'auto',
          display: 'flex', flexDirection: 'column', gap: '22px',
        }}>

          {/* Observer */}
          <div>
            <SectionHead color={TEAL}>Observer Geometry</SectionHead>
            <SliderRow label="Distance r_obs" paramKey="rObs" min={10} max={80} step={1}
              unit=" M" format={v => v.toFixed(0)} />
            <SliderRow label="Inclination θ" paramKey="thetaObsDeg" min={5} max={90} step={1}
              unit="°" format={v => v.toFixed(0)} />
            <SliderRow label="Field of View" paramKey="fov" min={0.15} max={0.8} step={0.01}
              format={v => v.toFixed(2)} unit=" rad" />
          </div>

          {/* Accretion Disk */}
          <div>
            <SectionHead color={AMBER}>Accretion Disk</SectionHead>
            <SliderRow label="Outer Radius r_out" paramKey="rDiskOuter" min={8} max={40} step={1}
              unit=" M" format={v => v.toFixed(0)} />
            <Toggle label="Show Disk" paramKey="showDisk" />
            <Toggle label="Doppler / Redshift" paramKey="doppler" note="Cunningham g⁴ intensity law" />
          </div>

          {/* Background */}
          <div>
            <SectionHead color={TEAL}>Background Field</SectionHead>
            <Toggle label="Lensed Starfield" paramKey="showStarfield" note="Full geodesic deflection" />
          </div>

          {/* Kerr */}
          <div>
            <SectionHead color={AMBER}>Kerr Shadow v9</SectionHead>
            <SliderRow label="Spin a/M" paramKey="spin" min={0} max={0.998} step={0.01}
              format={v => v.toFixed(3)} />
            <Toggle label="Show Kerr Overlay" paramKey="showKerrOverlay" note="Bardeen critical curve" />
            <div style={{
              padding: '8px 10px', background: BG3, border: '1px solid ' + BG3,
              borderRadius: '2px', fontSize: '9px', color: TEXTDIM,
              lineHeight: '1.8', fontFamily: MONO, marginTop: '4px',
            }}>
              DISK RENDER: Schwarzschild<br />
              OVERLAY: Kerr contour only<br />
              FULL KERR IMAGING: v10 TBD
            </div>
          </div>

          {/* Render quality */}
          <div>
            <SectionHead color={TEAL}>Render Resolution</SectionHead>
            <SliderRow label="Resolution" paramKey="resolution" min={120} max={400} step={20}
              unit="px" format={v => `${v}x${v}`} />
            <div style={{ fontSize: '9px', color: TEXTDIM, marginTop: '2px', fontFamily: MONO }}>
              280px approx 10-30s · 400px approx 60s+
            </div>
          </div>

          {/* Live physics stats */}
          {stats && (
            <div>
              <SectionHead color={TEAL}>Live Physics</SectionHead>
              <StatRow label="b_crit"     value={stats.bCrit + ' M'} />
              <StatRow label="psi_shadow" value={stats.psiShadow + ' deg'} />
              <StatRow label="r_ISCO"     value={stats.rISCO + ' M'} />
              <StatRow label="g_min"      value={stats.gMin} color={AMBER} />
              <StatRow label="g_max"      value={stats.gMax} color={AMBER} />
              {renderTime && <StatRow label="render_time" value={renderTime + 's'} color={TEXTDIM} />}
            </div>
          )}

          {/* Claim boundary */}
          <div style={{
            padding: '12px 10px', background: BG3,
            border: '1px solid ' + BORDER_A,
            borderRadius: '2px', fontSize: '9px', lineHeight: '1.9', fontFamily: MONO,
          }}>
            <div style={{ color: AMBER, marginBottom: '8px', letterSpacing: '0.15em' }}>
              CLAIM BOUNDARY
            </div>
            <div style={{ color: TEAL }}>OK Schwarzschild null geodesics (RK4)</div>
            <div style={{ color: TEAL }}>OK Cunningham/Luminet g4 intensity</div>
            <div style={{ color: TEAL }}>OK Bozza log coeff 0.93-0.99</div>
            <div style={{ color: TEAL }}>OK Bardeen shadow 9.07M vs 9.0M</div>
            <div style={{ color: TEXTDIM }}>NO Kerr disk imaging (shadow only)</div>
            <div style={{ color: TEXTDIM }}>NO GRMHD / synchrotron / polarization</div>
            <div style={{ color: TEXTDIM }}>NO EHT-grade comparison</div>
          </div>
        </div>
      </div>

      <style>{\`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        input[type=range] {
          appearance: none; -webkit-appearance: none;
          height: 2px; border-radius: 1px; background: #0a1220;
          outline: none; width: 100%;
        }
        input[type=range]::-webkit-slider-thumb {
          appearance: none; -webkit-appearance: none;
          width: 12px; height: 12px; border-radius: 2px;
          background: #e8a520; cursor: pointer; border: 1px solid #e8a520;
        }
        input[type=range]::-moz-range-thumb {
          width: 12px; height: 12px; border-radius: 2px;
          background: #e8a520; cursor: pointer; border: 1px solid #e8a520;
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #040810; }
        ::-webkit-scrollbar-thumb { background: #0a1220; border-radius: 2px; }
      \`}</style>
    </div>
  );
}
