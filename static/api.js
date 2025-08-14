// CHANGE THIS after deploying backend on Render:
const BACKEND = "https://backend-patient.onrender.com";

export async function api(path, opts = {}) {
  const token = localStorage.getItem("token");
  opts.headers = opts.headers || {};
  if (!(opts.body instanceof FormData)) {
    opts.headers["Content-Type"] = "application/json";
  }
  if (token) opts.headers["Authorization"] = `Bearer ${token}`;
  const r = await fetch(`${BACKEND}${path}`, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  const ct = r.headers.get("content-type") || "";
  return ct.includes("application/json") ? r.json() : r.blob();
}

export function setToken(t){ localStorage.setItem("token", t); }
export function clearToken(){ localStorage.removeItem("token"); }

