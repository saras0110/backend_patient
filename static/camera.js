export async function startCamera(videoEl) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  videoEl.srcObject = stream; await videoEl.play();
  return stream;
}

export function captureFrame(videoEl) {
  const c = document.createElement("canvas");
  c.width = videoEl.videoWidth; c.height = videoEl.videoHeight;
  const ctx = c.getContext("2d"); ctx.drawImage(videoEl, 0, 0);
  return new Promise((resolve)=> c.toBlob(b=> resolve(b), "image/jpeg", 0.85));
}
