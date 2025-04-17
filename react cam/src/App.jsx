import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import { FaceMesh } from "@mediapipe/face_mesh";
import * as camUtils from "@mediapipe/camera_utils";

function App() {
  const webcamRef = useRef(null);
  const [latestLandmarks, setLatestLandmarks] = useState(null);
  const [collectedLandmarks, setCollectedLandmarks] = useState([]);
  const cameraRef = useRef(null); // Keep camera instance for cleanup

  // Setup FaceMesh and capture loop
  useEffect(() => {
    const faceMesh = new FaceMesh({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });

    faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    faceMesh.onResults((results) => {
      if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        setLatestLandmarks(results.multiFaceLandmarks[0]);
      } else {
        setLatestLandmarks(null);
      }
    });

    const checkVideoReady = setInterval(() => {
      if (
        webcamRef.current &&
        webcamRef.current.video &&
        webcamRef.current.video.readyState === 4
      ) {
        // Start camera once video is ready
        clearInterval(checkVideoReady);
        cameraRef.current = new camUtils.Camera(webcamRef.current.video, {
          onFrame: async () => {
            await faceMesh.send({ image: webcamRef.current.video });
          },
          width: 640,
          height: 480,
        });
        cameraRef.current.start();
      }
    }, 500);

    const interval = setInterval(() => {
      if (latestLandmarks) {
        const timestamp = new Date().toISOString();
        const landmarksWithTime = latestLandmarks.map((point, index) => ({
          timestamp,
          index,
          x: point.x,
          y: point.y,
          z: point.z,
        }));
        setCollectedLandmarks((prev) => [...prev, ...landmarksWithTime]);
      }
    }, 5000);

    return () => {
      clearInterval(interval);
      if (cameraRef.current) cameraRef.current.stop();
    };
  }, [latestLandmarks]);

  const exportCSV = () => {
    if (collectedLandmarks.length === 0) {
      alert("❌ No landmarks collected yet.");
      return;
    }

    let csv = "timestamp,index,x,y,z\n";
    collectedLandmarks.forEach((point) => {
      csv += `${point.timestamp},${point.index},${point.x},${point.y},${point.z}\n`;
    });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "landmarks_collected.csv";
    a.click();
  };

  return (
    <div style={{ textAlign: "center", paddingTop: "40px" }}>
      {/* Hidden Webcam Component */}
      <Webcam
        ref={webcamRef}
        audio={false}
        mirrored={true}
        screenshotFormat="image/jpeg"
        style={{ display: "none" }}
      />
      <p>🔍 Capturing face landmarks in the background every 5 seconds...</p>
      <button onClick={exportCSV} style={buttonStyle}>
        📄 Export Collected Landmarks CSV
      </button>
    </div>
  );
}

const buttonStyle = {
  margin: "10px",
  padding: "10px 20px",
  fontSize: "16px",
  cursor: "pointer",
};

export default App;
