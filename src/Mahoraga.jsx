import { useState, useEffect, useRef, useCallback } from "react";

// ============================================================
// MAHORAGA — GPU geodesic integrator
// Fragment shader runs RK4 per pixel in parallel.
// Real-time Schwarzschild ray tracing with thin disk + lensed background.
// ============================================================

const VERTEX_SHADER = `
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

// Fragment shader implementing the same physics as v8 CPU code,
// but evaluated in parallel for every pixel.
const FRAGMENT_SHADER = `
precision highp float;
varying vec2 v_uv;

uniform vec2 u_resolution;
uniform float u_time;
uniform float u_rObs;
uniform float u_thetaObs;       // observer colatitude (rad)
uniform float u_phiObs;          // observer azimuth (for camera orbit)
uniform float u_fov;
uniform float u_rDiskInner;
uniform float u_rDiskOuter;
uniform float u_diskRotation;    // animation: disk material rotation angle
uniform float u_showDisk;
uniform float u_showStarfield;
uniform float u_doppler;
uniform float u_diskThickness;   // intensity falloff above/below plane

const float M = 1.0;
const float R_HORIZON = 2.0;
const float B_CRIT = 5.196152422706632;  // 3*sqrt(3)
const float PI = 3.14159265358979;
const float TAU = 6.28318530717958;
const int MAX_STEPS = 600;        // shader-affordable; CPU used 50000
const float DPHI = 0.025;          // larger step → fewer iterations, lensing still accurate

// ----- procedural hash for starfield -----
float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

vec3 sampleSky(float theta, float phi) {
  // Grid lines every 15°
  float g = PI / 12.0;
  float dPhi = min(mod(phi, g), g - mod(phi, g));
  float dTheta = min(mod(theta, g), g - mod(theta, g));
  bool onGrid = dPhi < 0.005 || dTheta < 0.005;

  // Stars (procedural, dense near grid intersections)
  float starSeed = hash(vec2(floor(theta * 400.0), floor(phi * 400.0)));
  vec3 col = vec3(0.0);
  if (starSeed > 0.9985) {
    float brightness = (starSeed - 0.9985) / 0.0015;
    col = vec3(0.9 + 0.1 * brightness, 0.85 + 0.15 * brightness, 0.75 + 0.25 * brightness) * (0.6 + 0.4 * brightness);
  } else if (onGrid) {
    col = vec3(
      phi > 0.0 ? 0.18 : 0.05,
      theta < PI * 0.5 ? 0.18 : 0.05,
      0.12
    );
  } else {
    float bg = 0.012 + 0.008 * sin(theta * 2.0 + phi * 0.7);
    col = vec3(bg, bg, bg + 0.005);
  }
  return col;
}

// Schwarzschild geodesic step in (u, du/dphi) where u = M/r
// d²u/dphi² = 3u² - u
void rk4Step(inout float u, inout float dudphi, float h) {
  float k1du = dudphi;
  float k1dv = 3.0 * u * u - u;
  float u2 = u + 0.5 * h * k1du;
  float v2 = dudphi + 0.5 * h * k1dv;
  float k2du = v2;
  float k2dv = 3.0 * u2 * u2 - u2;
  float u3 = u + 0.5 * h * k2du;
  float v3 = dudphi + 0.5 * h * k2dv;
  float k3du = v3;
  float k3dv = 3.0 * u3 * u3 - u3;
  float u4 = u + h * k3du;
  float v4 = dudphi + h * k3dv;
  float k4du = v4;
  float k4dv = 3.0 * u4 * u4 - u4;
  u = u + (h / 6.0) * (k1du + 2.0 * k2du + 2.0 * k3du + k4du);
  dudphi = dudphi + (h / 6.0) * (k1dv + 2.0 * k2dv + 2.0 * k3dv + k4dv);
}

// Disk emissivity + color ramp from observed temperature
vec3 colorRamp(float T) {
  T = clamp(T, 0.15, 1.4);
  if (T < 0.35) return mix(vec3(0.45, 0.04, 0.02), vec3(0.85, 0.22, 0.07), (T - 0.15) / 0.2);
  if (T < 0.55) return mix(vec3(0.85, 0.22, 0.07), vec3(1.0, 0.55, 0.22), (T - 0.35) / 0.2);
  if (T < 0.75) return mix(vec3(1.0, 0.55, 0.22), vec3(1.0, 0.9, 0.65), (T - 0.55) / 0.2);
  if (T < 0.95) return mix(vec3(1.0, 0.9, 0.65), vec3(0.97, 0.97, 0.97), (T - 0.75) / 0.2);
  return mix(vec3(0.97, 0.97, 0.97), vec3(0.62, 0.82, 1.08), (T - 0.95) / 0.45);
}

// Soft falloff for disk thickness (so it doesn't look infinitely thin)
float diskProfile(float r, float z) {
  if (r < u_rDiskInner || r > u_rDiskOuter) return 0.0;
  float radial = pow(u_rDiskInner / r, 1.5);
  float vert = exp(-abs(z) * 6.0 / u_diskThickness);
  return radial * vert;
}

void main() {
  // Pixel direction in screen coords
  vec2 uv = (v_uv - 0.5) * 2.0;
  uv.x *= u_resolution.x / u_resolution.y;
  vec2 pixel = uv * u_fov;
  float psi = length(pixel);
  float azimuth = atan(pixel.y, pixel.x);

  // Observer setup
  float sinTheta = sin(u_thetaObs);
  float cosTheta = cos(u_thetaObs);
  float sinPhi = sin(u_phiObs);
  float cosPhi = cos(u_phiObs);

  // Observer position
  vec3 O = u_rObs * vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);

  // Local frame
  vec3 er = O / u_rObs;
  vec3 etheta = vec3(cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta);
  vec3 ephi = vec3(-sinPhi, cosPhi, 0.0);

  // Photon direction (initial, in observer frame)
  vec3 d = normalize(-er + pixel.y * etheta + pixel.x * ephi);

  // Impact parameter (local-observer formula)
  float cosPsi = -dot(d, er);
  float psiAngle = acos(clamp(cosPsi, -1.0, 1.0));
  float b = u_rObs * sin(psiAngle) / sqrt(1.0 - 2.0 * M / u_rObs);

  // === SHADOW CHECK ===
  if (b <= B_CRIT) {
    // Pure shadow
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // Orbital plane normal (O x d) — this is conserved for the photon
  vec3 n = normalize(cross(O, d));

  // In-plane basis: e1 along observer position, e2 perpendicular in motion sense
  vec3 e1 = er;
  vec3 e2 = cross(n, e1);
  if (dot(e2, d) < 0.0) e2 = -e2;

  // Equator (z=0) intersects orbital plane along direction (z x n)
  vec3 zCrossN = vec3(-n.y, n.x, 0.0);
  float zcrLen = length(zCrossN);
  float phiEq = 0.0;
  bool hasEquator = zcrLen > 1e-4;
  if (hasEquator) {
    zCrossN /= zcrLen;
    float eq1 = dot(zCrossN, e1);
    float eq2 = dot(zCrossN, e2);
    phiEq = atan(eq2, eq1);
  }

  // === GEODESIC INTEGRATION ===
  // Initial conditions at observer
  float u = M / u_rObs;
  float rhs = 1.0 / (b * b) - u * u * (1.0 - 2.0 * u);
  if (rhs < 0.0) { gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0); return; }
  float dudphi = sqrt(rhs);
  float phi = 0.0;
  bool turned = false;
  bool diskHit = false;
  float diskHitR = 0.0;
  float diskHitPhi = 0.0;
  vec3 diskHitPos = vec3(0.0);
  bool captured = false;
  bool escaped = false;

  // Track sign of (phi - phiEq) for equator crossing detection
  float prevSinPhi = sin(0.0 - phiEq);
  float prevPhi = 0.0;
  float prevR = u_rObs;

  for (int step = 0; step < MAX_STEPS; step++) {
    rk4Step(u, dudphi, DPHI);
    phi += DPHI;
    float r = M / u;

    if (r <= R_HORIZON * 1.001) {
      captured = true;
      break;
    }

    if (!turned && dudphi < 0.0) turned = true;
    if (turned && u <= M / u_rObs) {
      escaped = true;
      break;
    }

    // Equator crossing check (only if we have disk and equator is meaningful)
    if (u_showDisk > 0.5 && hasEquator && !diskHit) {
      float curSinPhi = sin(phi - phiEq);
      // Also check the other side of the equator line (phi - phiEq - PI)
      float curSinPhi2 = sin(phi - phiEq - PI);
      float prevSinPhi2 = sin(prevPhi - phiEq - PI);

      // Sign change in either branch → crossing
      bool crossed = (prevSinPhi * curSinPhi < 0.0 && abs(curSinPhi - prevSinPhi) < 0.5) ||
                     (prevSinPhi2 * curSinPhi2 < 0.0 && abs(curSinPhi2 - prevSinPhi2) < 0.5);

      if (crossed && r >= u_rDiskInner * 0.95 && r <= u_rDiskOuter * 1.05) {
        // Linear interp to find crossing radius
        float useSinPrev = (prevSinPhi * curSinPhi < 0.0) ? prevSinPhi : prevSinPhi2;
        float useSinCur = (prevSinPhi * curSinPhi < 0.0) ? curSinPhi : curSinPhi2;
        float tFrac = -useSinPrev / (useSinCur - useSinPrev);
        float rHit = prevR + tFrac * (r - prevR);
        float phiHit = prevPhi + tFrac * DPHI;
        if (rHit >= u_rDiskInner && rHit <= u_rDiskOuter) {
          diskHit = true;
          diskHitR = rHit;
          diskHitPhi = phiHit;
          // Reconstruct 3D position
          diskHitPos = rHit * (cos(phiHit) * e1 + sin(phiHit) * e2);
          break;
        }
      }
      prevSinPhi = curSinPhi;
    }
    prevPhi = phi;
    prevR = r;
  }

  vec3 color = vec3(0.0);

  // === DISK SHADING ===
  if (diskHit) {
    float rDisk = length(diskHitPos.xy);
    float phiDisk = atan(diskHitPos.y, diskHitPos.x);

    // Doppler / gravitational redshift (Cunningham)
    float bZ = u_doppler > 0.5 ? b * n.z : 0.0;
    float Omega = sqrt(M / (rDisk * rDisk * rDisk));
    float gFactor = sqrt(max(0.001, 1.0 - 3.0 * M / rDisk)) / (1.0 + Omega * bZ);
    gFactor = clamp(gFactor, 0.05, 3.0);

    // Disk emission (Shakura-Sunyaev temperature profile)
    float localT = pow(u_rDiskInner / rDisk, 0.75);
    float obsT = localT * gFactor;
    float Iemit = pow(u_rDiskInner / rDisk, 2.0);
    float Iobs = pow(gFactor, 4.0) * Iemit;

    // Add procedural turbulence to the disk (advected with rotation)
    float advPhi = phiDisk - u_diskRotation * Omega * 20.0;
    float turb = 0.7 + 0.3 * sin(advPhi * 8.0 + rDisk * 0.5) * sin(advPhi * 13.0 - rDisk * 0.3);

    vec3 cBase = colorRamp(obsT);
    color = cBase * min(1.2, Iobs * 8.5) * turb;
  }
  // === STARFIELD BACKGROUND ===
  else if (escaped && u_showStarfield > 0.5) {
    // Compute total deflection: alpha = (phi_swept - 2*phi_peri_to_obs)
    // For our trace from observer-to-observer, this equals (phi - 2*pi)... 
    // Use the simpler approach: lookup-style, treat phi as direct
    float skyTheta = PI - psiAngle;
    // Use accumulated phi to figure out the sky direction
    // The photon came from direction (psi + deflection) from BH center
    float deflection = phi - PI;
    skyTheta = PI - (psiAngle + deflection);
    skyTheta = mod(skyTheta + TAU, TAU);
    if (skyTheta > PI) skyTheta = TAU - skyTheta;
    color = sampleSky(skyTheta, azimuth);
  }
  else if (captured) {
    color = vec3(0.0);
  }

  // Gentle tonemapping
  color = color / (1.0 + color * 0.4);
  gl_FragColor = vec4(color, 1.0);
}
`;

// ============================================================
// React component
// ============================================================
export default function AstroNovaMahoraga() {
  const canvasRef = useRef(null);
  const glRef = useRef(null);
  const programRef = useRef(null);
  const uniformsRef = useRef({});
  const animRef = useRef(null);
  const startTimeRef = useRef(performance.now());
  const [fps, setFps] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [glError, setGlError] = useState(null);

  const [params, setParams] = useState({
    rObs: 25,
    thetaObsDeg: 80,
    fov: 0.5,
    rDiskOuter: 18,
    diskThickness: 0.3,
    cameraOrbit: true,
    orbitSpeed: 0.08,
    showDisk: true,
    showStarfield: true,
    doppler: true,
    quality: 1.0,  // resolution multiplier
  });

  const paramsRef = useRef(params);
  useEffect(() => { paramsRef.current = params; }, [params]);

  const compileShader = (gl, type, source) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, source);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(s);
      gl.deleteShader(s);
      throw new Error(`Shader compile error: ${err}`);
    }
    return s;
  };

  // Initialize WebGL once
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl", { preserveDrawingBuffer: true }) ||
               canvas.getContext("experimental-webgl");
    if (!gl) {
      setGlError("WebGL not supported by this browser.");
      return;
    }
    glRef.current = gl;

    try {
      const vs = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
      const fs = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
      const prog = gl.createProgram();
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(prog));
      }
      programRef.current = prog;

      // Quad
      const buf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
        -1, -1,  1, -1, -1, 1,
        -1,  1,  1, -1,  1, 1
      ]), gl.STATIC_DRAW);
      const posLoc = gl.getAttribLocation(prog, "a_position");
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

      uniformsRef.current = {
        u_resolution: gl.getUniformLocation(prog, "u_resolution"),
        u_time: gl.getUniformLocation(prog, "u_time"),
        u_rObs: gl.getUniformLocation(prog, "u_rObs"),
        u_thetaObs: gl.getUniformLocation(prog, "u_thetaObs"),
        u_phiObs: gl.getUniformLocation(prog, "u_phiObs"),
        u_fov: gl.getUniformLocation(prog, "u_fov"),
        u_rDiskInner: gl.getUniformLocation(prog, "u_rDiskInner"),
        u_rDiskOuter: gl.getUniformLocation(prog, "u_rDiskOuter"),
        u_diskRotation: gl.getUniformLocation(prog, "u_diskRotation"),
        u_showDisk: gl.getUniformLocation(prog, "u_showDisk"),
        u_showStarfield: gl.getUniformLocation(prog, "u_showStarfield"),
        u_doppler: gl.getUniformLocation(prog, "u_doppler"),
        u_diskThickness: gl.getUniformLocation(prog, "u_diskThickness"),
      };
    } catch (err) {
      setGlError(err.message);
      return;
    }
  }, []);

  // Animation loop
  useEffect(() => {
    const gl = glRef.current;
    const prog = programRef.current;
    const canvas = canvasRef.current;
    if (!gl || !prog || !canvas) return;

    let lastT = performance.now();
    let frameCount = 0;
    let lastFpsT = lastT;
    let phiObs = 0;

    const loop = (t) => {
      const dt = (t - lastT) / 1000;
      lastT = t;

      const p = paramsRef.current;

      // Resize handling
      const dpr = Math.min(window.devicePixelRatio || 1, 2) * p.quality;
      const cw = canvas.clientWidth;
      const ch = canvas.clientHeight;
      const w = Math.max(64, Math.floor(cw * dpr));
      const h = Math.max(64, Math.floor(ch * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      gl.viewport(0, 0, w, h);

      gl.useProgram(prog);

      if (p.cameraOrbit && playing) phiObs += p.orbitSpeed * dt;
      const tNow = (t - startTimeRef.current) / 1000;

      gl.uniform2f(uniformsRef.current.u_resolution, w, h);
      gl.uniform1f(uniformsRef.current.u_time, tNow);
      gl.uniform1f(uniformsRef.current.u_rObs, p.rObs);
      gl.uniform1f(uniformsRef.current.u_thetaObs, (p.thetaObsDeg * Math.PI) / 180);
      gl.uniform1f(uniformsRef.current.u_phiObs, phiObs);
      gl.uniform1f(uniformsRef.current.u_fov, p.fov);
      gl.uniform1f(uniformsRef.current.u_rDiskInner, 6.0);
      gl.uniform1f(uniformsRef.current.u_rDiskOuter, p.rDiskOuter);
      gl.uniform1f(uniformsRef.current.u_diskRotation, playing ? tNow : 0);
      gl.uniform1f(uniformsRef.current.u_showDisk, p.showDisk ? 1 : 0);
      gl.uniform1f(uniformsRef.current.u_showStarfield, p.showStarfield ? 1 : 0);
      gl.uniform1f(uniformsRef.current.u_doppler, p.doppler ? 1 : 0);
      gl.uniform1f(uniformsRef.current.u_diskThickness, p.diskThickness);

      gl.drawArrays(gl.TRIANGLES, 0, 6);

      frameCount++;
      if (t - lastFpsT > 500) {
        setFps(Math.round((frameCount * 1000) / (t - lastFpsT)));
        frameCount = 0;
        lastFpsT = t;
      }

      animRef.current = requestAnimationFrame(loop);
    };

    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [playing]);

  const updateParam = (k, v) => setParams(p => ({ ...p, [k]: v }));

  const exportPNG = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = `mahoraga_r${params.rObs}_i${params.thetaObsDeg}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  // ===== Style helpers =====
  const Slider = ({ label, paramKey, min, max, step, format, unit }) => (
    <div style={{ marginBottom: "11px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
        <span style={{ color: "#8b9bb4", fontSize: "10px", letterSpacing: "0.12em", textTransform: "uppercase" }}>
          {label}
        </span>
        <span style={{ color: "#e2e8f0", fontSize: "11px", fontFamily: "'JetBrains Mono', monospace" }}>
          {format ? format(params[paramKey]) : params[paramKey]}{unit || ""}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step}
        value={params[paramKey]}
        onChange={e => updateParam(paramKey, parseFloat(e.target.value))}
        style={{ width: "100%" }} />
    </div>
  );

  const Toggle = ({ label, paramKey, note }) => (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "10px" }}>
      <div style={{ flex: 1 }}>
        <span style={{ color: "#8b9bb4", fontSize: "10px", letterSpacing: "0.12em", textTransform: "uppercase" }}>{label}</span>
        {note && <div style={{ color: "#475569", fontSize: "9px", marginTop: "2px" }}>{note}</div>}
      </div>
      <div onClick={() => updateParam(paramKey, !params[paramKey])}
        style={{
          width: "30px", height: "16px", borderRadius: "9px",
          background: params[paramKey] ? "#ec4899" : "#1e293b",
          position: "relative", cursor: "pointer", transition: "background 0.15s",
          flexShrink: 0
        }}>
        <div style={{
          position: "absolute", top: "2px",
          left: params[paramKey] ? "16px" : "2px",
          width: "12px", height: "12px", borderRadius: "50%",
          background: "white", transition: "left 0.15s"
        }} />
      </div>
    </div>
  );

  if (glError) {
    return (
      <div style={{ background: "#000", color: "#f00", padding: "40px", fontFamily: "monospace" }}>
        WebGL error: {glError}
      </div>
    );
  }

  return (
    <div style={{
      width: "100%", height: "100vh", background: "#000",
      fontFamily: "'JetBrains Mono', 'SF Mono', monospace",
      color: "#e2e8f0", display: "flex", overflow: "hidden", position: "relative"
    }}>

      {/* Canvas — full bleed */}
      <div style={{ flex: 1, position: "relative", background: "#000" }}>
        <canvas ref={canvasRef}
          style={{
            width: "100%", height: "100%", display: "block",
            cursor: "grab"
          }}
        />

        {/* HUD overlay top-left */}
        <div style={{
          position: "absolute", top: "20px", left: "20px",
          display: "flex", flexDirection: "column", gap: "8px", pointerEvents: "none"
        }}>
          <div style={{
            display: "flex", alignItems: "center", gap: "10px",
            padding: "8px 14px",
            background: "rgba(0,0,0,0.5)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(236,72,153,0.15)",
            borderRadius: "2px"
          }}>
            <div style={{
              width: "8px", height: "8px", borderRadius: "50%",
              background: "#ec4899", boxShadow: "0 0 8px #ec4899",
              animation: "pulse 1.4s ease-in-out infinite"
            }} />
            <span style={{ fontSize: "11px", letterSpacing: "0.2em", color: "#f1f5f9" }}>
              MAHORAGA · LIVE
            </span>
          </div>
          <div style={{
            padding: "6px 14px",
            background: "rgba(0,0,0,0.5)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(255,255,255,0.05)",
            fontSize: "10px", color: "#64748b", letterSpacing: "0.1em"
          }}>
            {fps} FPS · GPU GEODESICS
          </div>
        </div>

        {/* HUD bottom — claim boundary */}
        <div style={{
          position: "absolute", bottom: "20px", left: "20px",
          padding: "8px 16px",
          background: "rgba(0,0,0,0.5)",
          backdropFilter: "blur(12px)",
          border: "1px solid rgba(255,255,255,0.05)",
          fontSize: "9px", color: "#475569", letterSpacing: "0.12em",
          pointerEvents: "none"
        }}>
          SCHWARZSCHILD RK4 · CUNNINGHAM g⁴ · REAL-TIME
        </div>

        {/* Playback controls bottom center */}
        <div style={{
          position: "absolute", bottom: "20px", left: "50%",
          transform: "translateX(-50%)",
          display: "flex", gap: "8px"
        }}>
          <button onClick={() => setPlaying(p => !p)} style={{
            background: "rgba(0,0,0,0.6)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(236,72,153,0.3)",
            color: "#ec4899", padding: "8px 18px", borderRadius: "2px",
            cursor: "pointer", fontSize: "11px", letterSpacing: "0.15em",
            fontFamily: "inherit", fontWeight: "600"
          }}>
            {playing ? "⏸ PAUSE" : "▶ PLAY"}
          </button>
          <button onClick={exportPNG} style={{
            background: "rgba(0,0,0,0.6)",
            backdropFilter: "blur(12px)",
            border: "1px solid rgba(255,255,255,0.1)",
            color: "#94a3b8", padding: "8px 18px", borderRadius: "2px",
            cursor: "pointer", fontSize: "11px", letterSpacing: "0.15em",
            fontFamily: "inherit"
          }}>
            CAPTURE FRAME
          </button>
        </div>
      </div>

      {/* Control panel right side */}
      <div style={{
        width: "260px",
        background: "rgba(5,5,8,0.85)",
        backdropFilter: "blur(16px)",
        borderLeft: "1px solid rgba(255,255,255,0.04)",
        padding: "24px 20px",
        overflowY: "auto",
        display: "flex", flexDirection: "column", gap: "22px"
      }}>

        {/* Brand */}
        <div>
          <div style={{
            fontSize: "9px", letterSpacing: "0.3em",
            color: "#ec4899", marginBottom: "6px"
          }}>ASTRONOVA ENGINE</div>
          <div style={{ fontSize: "18px", fontWeight: "700", color: "#f1f5f9", letterSpacing: "0.05em" }}>
            MAHORAGA
          </div>
          <div style={{ fontSize: "10px", color: "#475569", marginTop: "4px", lineHeight: "1.5" }}>
            Real-time GPU geodesic renderer.<br />
            Fragment-shader RK4 per pixel.
          </div>
        </div>

        {/* Camera */}
        <div>
          <SectionHeader>Camera</SectionHeader>
          <Slider label="Distance" paramKey="rObs" min={8} max={60} step={0.5}
            unit=" M" format={v => v.toFixed(1)} />
          <Slider label="Inclination" paramKey="thetaObsDeg" min={5} max={90} step={1}
            unit="°" format={v => v.toFixed(0)} />
          <Slider label="Field of View" paramKey="fov" min={0.2} max={0.9} step={0.01}
            format={v => v.toFixed(2)} unit=" rad" />
          <Toggle label="Auto-Orbit" paramKey="cameraOrbit" />
          <Slider label="Orbit Speed" paramKey="orbitSpeed" min={0} max={0.5} step={0.01}
            format={v => v.toFixed(2)} />
        </div>

        {/* Disk */}
        <div>
          <SectionHeader>Accretion Disk</SectionHeader>
          <Slider label="Outer Radius" paramKey="rDiskOuter" min={8} max={40} step={0.5}
            unit=" M" format={v => v.toFixed(1)} />
          <Slider label="Disk Thickness" paramKey="diskThickness" min={0.1} max={1.2} step={0.05}
            format={v => v.toFixed(2)} />
          <Toggle label="Show Disk" paramKey="showDisk" />
          <Toggle label="Doppler / Redshift" paramKey="doppler" note="Cunningham g⁴" />
        </div>

        {/* Background */}
        <div>
          <SectionHeader>Background</SectionHeader>
          <Toggle label="Lensed Starfield" paramKey="showStarfield" />
        </div>

        {/* Quality */}
        <div>
          <SectionHeader>Render</SectionHeader>
          <Slider label="Resolution Scale" paramKey="quality" min={0.4} max={1.5} step={0.1}
            format={v => `${(v*100).toFixed(0)}%`} />
          <div style={{ fontSize: "9px", color: "#334155", marginTop: "4px", lineHeight: "1.5" }}>
            Lower for smoother fps on weaker GPUs.
          </div>
        </div>

        {/* Physics info */}
        <div style={{
          marginTop: "auto", padding: "12px",
          background: "rgba(0,0,0,0.4)",
          border: "1px solid rgba(236,72,153,0.1)",
          borderRadius: "2px",
          fontSize: "9px", color: "#475569", lineHeight: "1.8", letterSpacing: "0.05em"
        }}>
          <div style={{ color: "#94a3b8", marginBottom: "6px", letterSpacing: "0.15em", fontSize: "9px" }}>
            PHYSICS · LIVE
          </div>
          b_crit = 3√3 M = 5.196<br />
          r_ISCO = 6 M<br />
          R(u): d²u/dφ² = 3u² − u<br />
          g = √(1−3M/r) / (1+Ωb_z)<br />
          I_obs = g⁴ · I_emit
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        input[type=range] {
          appearance: none; -webkit-appearance: none;
          height: 2px; border-radius: 1px;
          background: linear-gradient(to right, #ec4899, #1e293b);
          outline: none;
        }
        input[type=range]::-webkit-slider-thumb {
          appearance: none; -webkit-appearance: none;
          width: 12px; height: 12px; border-radius: 50%;
          background: #f1f5f9; cursor: pointer;
          box-shadow: 0 0 8px rgba(236,72,153,0.4);
        }
        input[type=range]::-moz-range-thumb {
          width: 12px; height: 12px; border-radius: 50%;
          background: #f1f5f9; cursor: pointer; border: none;
          box-shadow: 0 0 8px rgba(236,72,153,0.4);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
        button:hover { filter: brightness(1.2); }
      `}</style>
    </div>
  );
}

const SectionHeader = ({ children }) => (
  <div style={{
    fontSize: "9px", color: "#ec4899", letterSpacing: "0.3em",
    marginBottom: "14px", textTransform: "uppercase",
    paddingBottom: "8px", borderBottom: "1px solid rgba(236,72,153,0.1)"
  }}>
    {children}
  </div>
);
