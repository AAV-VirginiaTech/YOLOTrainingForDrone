// webcam_js_code.js
var video;
var div = null;
var stream;
var captureCanvas;
var imgElement;
var labelElement;
var pendingResolve = null;
var shutdown = false;

function removeDom() {
  if (stream) stream.getVideoTracks()[0].stop();
  if (video) video.remove();
  if (div) div.remove();
  video = null;
  div = null;
  stream = null;
  imgElement = null;
  captureCanvas = null;
  labelElement = null;
}

function onAnimationFrame() {
  if (!shutdown) window.requestAnimationFrame(onAnimationFrame);
  if (pendingResolve) {
    var result = "";
    if (!shutdown) {
      captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
      result = captureCanvas.toDataURL('image/jpeg', 0.8);
    }
    var lp = pendingResolve;
    pendingResolve = null;
    lp(result);
  }
}

async function createDom() {
  if (div !== null) return stream;

  div = document.createElement('div');
  div.style.border = '2px solid black';
  div.style.padding = '3px';
  div.style.width = '100%';
  div.style.maxWidth = '600px';
  document.body.appendChild(div);

  const modelOut = document.createElement('div');
  modelOut.innerHTML = "<span>Status:</span>";
  labelElement = document.createElement('span');
  labelElement.innerText = 'No data';
  labelElement.style.fontWeight = 'bold';
  modelOut.appendChild(labelElement);
  div.appendChild(modelOut);

  video = document.createElement('video');
  video.style.display = 'block';
  video.width = div.clientWidth - 6;
  video.setAttribute('playsinline', '');
  video.onclick = () => { shutdown = true; };
  stream = await navigator.mediaDevices.getUserMedia({video: { facingMode: "environment"}});
  div.appendChild(video);

  imgElement = document.createElement('img');
  imgElement.style.position = 'absolute';
  imgElement.style.zIndex = 1;
  imgElement.style.pointerEvents = 'none'; // so clicks go to the video
  div.appendChild(imgElement);

  const instruction = document.createElement('div');
  instruction.innerHTML =
    '<span style="color: red; font-weight: bold;">Click the video to stop</span>';
  div.appendChild(instruction);
  instruction.onclick = () => { shutdown = true; };

  video.srcObject = stream;
  await video.play();

  captureCanvas = document.createElement('canvas');
  captureCanvas.width = 640;
  captureCanvas.height = 480;

  window.requestAnimationFrame(onAnimationFrame);
  return stream;
}

async function stream_frame(label, imgData) {
  if (shutdown) {
    removeDom();
    shutdown = false;
    return '';
  }

  stream = await createDom();

  if (label != "") {
    labelElement.innerHTML = label;
  }

  if (imgData != "") {
    // Position overlay exactly over the video element
    var videoRect = video.getClientRects()[0];
    imgElement.style.top = videoRect.top + "px";
    imgElement.style.left = videoRect.left + "px";
    imgElement.style.width = videoRect.width + "px";
    imgElement.style.height = videoRect.height + "px";
    imgElement.src = imgData;
  }

  var result = await new Promise(resolve => { pendingResolve = resolve; });
  shutdown = false;

  return {
    'img': result
  };
}
