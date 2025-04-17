import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import Papa from "papaparse";

const FaceCapture = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const [capturing, setCapturing] = useState(false);
  const [status, setStatus] = useState("Idle");
  const [landmarkData, setLandmarkData] = useState([]);

  useEffect(() => {
    return () => clearInterval(intervalRef.current); // Clean up
  }, []);

  const drawLandmarks = (landmarks) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = "cyan";
    landmarks.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const captureLandmarks = async (detector) => {
    const video = webcamRef.current.video;
    if (video.readyState === 4) {
      const faces = await detector.estimateFaces(video);
      if (faces.length > 0) {
        const landmarks = faces[0].keypoints;
        drawLandmarks(landmarks);

        // const flat = landmarks.flatMap((pt) => [
        //   pt.x.toFixed(2),
        //   pt.y.toFixed(2),
        //   pt.z?.toFixed(2) || "0"
        // ]);
        // const flat = landmarks.flatMap((pt) => [
        //   ((pt.x / 640) * 2 - 1).toFixed(5),
        //   ((pt.y / 480) * 2 - 1).toFixed(5),
        //   (pt.z?.toFixed(5) || "0")
        // ]);
        // 
        const flat = landmarks.flatMap((pt) => [
          ((pt.x / video.videoWidth) * 2 - 1).toFixed(5),   // normalized X
          ((pt.y / video.videoHeight) * 2 - 1).toFixed(5),  // normalized Y
          (pt.z !== undefined
            ? ((pt.z / video.videoWidth) * 2 - 1).toFixed(5)
            : "0.00000")                                     // normalized Z
        ]);
        
        
        
        
        setLandmarkData((prev) => [...prev, flat]);
        setStatus("Face detected");
      } else {
        setStatus("No face detected");
      }
    }
  };

  const startCapture = async () => {
    setCapturing(true);
    setStatus("Loading model...");

    await tf.setBackend("webgl");
    await tf.ready();

    const detector = await faceLandmarksDetection.createDetector(
      faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
      {
        runtime: "tfjs",
        refineLandmarks: true,
      }
    );

    setStatus("Model loaded. Capturing...");
    intervalRef.current = setInterval(() => {
      captureLandmarks(detector);
    }, 5000);
  };

  const stopCapture = () => {
    setCapturing(false);
    clearInterval(intervalRef.current);
    setStatus("Stopped");

    if (landmarkData.length === 0) {
      alert("No facial landmarks were captured.");
      return;
    }

    const headers = Array.from({ length: 468 * 3 }, (_, i) => `val_${i + 1}`);
    const csv = Papa.unparse({ fields: headers, data: landmarkData });

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "facial_landmarks.csv";
    a.click();
    URL.revokeObjectURL(url);

    setLandmarkData([]);
  };

  return (
    <div className="flex flex-col items-center p-4">
      <h2 className="text-xl font-bold mb-4">Facial Landmark Capture</h2>

      {capturing && (
        <>
          <Webcam
            ref={webcamRef}
            audio={false}
            width={640}
            height={480}
            mirrored
            videoConstraints={{ facingMode: "user", width: 640, height: 480 }}
            className="absolute opacity-0"
          />
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="rounded shadow-md"
          />
        </>
      )}

      {!capturing && <div className="h-[480px] w-[640px] bg-gray-200 rounded flex items-center justify-center">Camera Off</div>}

      <p className="mt-4 text-sm text-gray-600">Status: {status}</p>
      <p className="text-sm text-gray-600">
        Captured Frames: {landmarkData.length}
      </p>

      <div className="mt-4 space-x-4">
        {!capturing ? (
          <button
            onClick={startCapture}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
          >
            Start
          </button>
        ) : (
          <button
            onClick={stopCapture}
            className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
          >
            Stop & Save CSV
          </button>
        )}
      </div>
    </div>
  );
};

export default FaceCapture;
