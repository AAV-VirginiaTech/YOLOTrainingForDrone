// webcam_js_code.js
// This code provides webcam streaming functionality for real-time video capture and display
// It's designed to work with Python backends for image processing (e.g., YOLO model inference)

// Global variables to manage video stream and DOM elements
var video;              // Video element that displays the webcam feed
var div = null;         // Container div for all webcam UI elements
var stream;             // MediaStream object from getUserMedia API
var captureCanvas;      // Canvas element used to capture video frames as images
var imgElement;         // Image element for displaying overlay (e.g., detection results)
var labelElement;       // Span element for displaying status/detection labels
var pendingResolve = null;  // Promise resolver for frame capture synchronization
var shutdown = false;   // Flag to control stream shutdown

/**
 * Cleanup function to stop the webcam stream and remove all DOM elements
 * Stops video tracks and resets all global variables to null
 */
function removeDom() {
  if (stream) stream.getVideoTracks()[0].stop();  // Stop the webcam stream
  if (video) video.remove();    // Remove video element from DOM
  if (div) div.remove();        // Remove container div from DOM
  // Reset all global variables
  video = null;
  div = null;
  stream = null;
  imgElement = null;
  captureCanvas = null;
  labelElement = null;
}

/**
 * Animation frame callback that captures video frames continuously
 * Uses requestAnimationFrame for smooth, efficient frame capture
 * When a frame is requested (pendingResolve is set), it draws the current
 * video frame to a canvas and converts it to a JPEG data URL
 */
function onAnimationFrame() {
  // Continue animation loop unless shutdown is triggered
  if (!shutdown) window.requestAnimationFrame(onAnimationFrame);
  
  // Check if there's a pending request for a frame
  if (pendingResolve) {
    var result = "";
    if (!shutdown) {
      // Draw current video frame to canvas (640x480 resolution)
      captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
      // Convert canvas to JPEG data URL with 80% quality
      result = captureCanvas.toDataURL('image/jpeg', 0.8);
    }
    // Resolve the promise with the captured frame data
    var lp = pendingResolve;
    pendingResolve = null;
    lp(result);
  }
}

/**
 * Creates and initializes all DOM elements needed for webcam display
 * Sets up video element, overlay image, status label, and instructions
 * Requests access to the user's camera (rear-facing if available)
 * @returns {MediaStream} The webcam media stream
 */
async function createDom() {
  // Return existing stream if already initialized
  if (div !== null) return stream;

  // Create main container div with styling
  div = document.createElement('div');
  div.style.border = '2px solid black';
  div.style.padding = '3px';
  div.style.width = '100%';
  div.style.maxWidth = '600px';
  document.body.appendChild(div);

  // Create status display section
  const modelOut = document.createElement('div');
  modelOut.innerHTML = "<span>Status:</span>";
  labelElement = document.createElement('span');
  labelElement.innerText = 'No data';
  labelElement.style.fontWeight = 'bold';
  modelOut.appendChild(labelElement);
  div.appendChild(modelOut);

  // Create and configure video element
  video = document.createElement('video');
  video.style.display = 'block';
  video.width = div.clientWidth - 6;  // Account for border and padding
  video.setAttribute('playsinline', '');  // Required for iOS Safari
  video.onclick = () => { shutdown = true; };  // Click video to stop stream
  
  // Request camera access (prefer rear-facing camera if available)
  stream = await navigator.mediaDevices.getUserMedia({video: { facingMode: "environment"}});
  div.appendChild(video);

  // Create overlay image element for displaying detection results
  imgElement = document.createElement('img');
  imgElement.style.position = 'absolute';
  imgElement.style.zIndex = 1;  // Ensure overlay appears above video
  imgElement.style.pointerEvents = 'none'; // Allow clicks to pass through to video
  div.appendChild(imgElement);

  // Create instruction text for users
  const instruction = document.createElement('div');
  instruction.innerHTML =
    '<span style="color: red; font-weight: bold;">Click the video to stop</span>';
  div.appendChild(instruction);
  instruction.onclick = () => { shutdown = true; };  // Click instruction to stop

  // Connect stream to video element and start playback
  video.srcObject = stream;
  await video.play();

  // Create canvas for capturing video frames
  captureCanvas = document.createElement('canvas');
  captureCanvas.width = 640;
  captureCanvas.height = 480;

  // Start the animation frame loop for continuous frame capture
  window.requestAnimationFrame(onAnimationFrame);
  return stream;
}

/**
 * Main function called from Python to stream video frames
 * Updates the display with labels and overlay images, then captures and returns the next frame
 * @param {string} label - Status or detection label to display (e.g., "Detected: car")
 * @param {string} imgData - Base64 image data URL for overlay (detection visualization)
 * @returns {Promise<Object>} Object containing 'img' property with captured frame data URL
 */
async function stream_frame(label, imgData) {
  // Handle shutdown: cleanup and reset
  if (shutdown) {
    removeDom();
    shutdown = false;
    return '';
  }

  // Initialize DOM elements and start webcam stream
  stream = await createDom();

  // Update status label if provided
  if (label != "") {
    labelElement.innerHTML = label;
  }

  // Update overlay image if provided (e.g., bounding boxes from YOLO)
  if (imgData != "") {
    // Position overlay to exactly match the video element dimensions
    var videoRect = video.getClientRects()[0];
    imgElement.style.top = videoRect.top + "px";
    imgElement.style.left = videoRect.left + "px";
    imgElement.style.width = videoRect.width + "px";
    imgElement.style.height = videoRect.height + "px";
    imgElement.src = imgData;  // Set overlay image source
  }

  // Wait for the next frame to be captured by onAnimationFrame
  var result = await new Promise(resolve => { pendingResolve = resolve; });
  shutdown = false;

  // Return captured frame as data URL
  return {
    'img': result
  };
}