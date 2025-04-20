import React, { useRef, useEffect } from "react";
import { FaceMesh } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";
import * as drawingUtils from "@mediapipe/drawing_utils";
import { FACEMESH_TESSELATION } from "@mediapipe/face_mesh";

const EmotionDetector = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (
      typeof window !== "undefined" &&
      videoRef.current &&
      canvasRef.current
    ) {
      const videoElement = videoRef.current;
      const canvasElement = canvasRef.current;
      const canvasCtx = canvasElement.getContext("2d");

      const faceMesh = new FaceMesh({
        locateFile: (file) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.3,
        minTrackingConfidence: 0.3,
      });

      faceMesh.onResults((results) => {
        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.drawImage(
          results.image,
          0,
          0,
          canvasElement.width,
          canvasElement.height
        );

        if (
          results.multiFaceLandmarks &&
          results.multiFaceLandmarks.length > 0
        ) {
          const landmarks = results.multiFaceLandmarks[0];

          // Draw mesh
          drawingUtils.drawConnectors(
            canvasCtx,
            landmarks,
            FACEMESH_TESSELATION,
            { color: "#00FF00", lineWidth: 0.5 }
          );

          const landmarkData = landmarks.slice(0, 468).map((pt) => [pt.x, pt.y, pt.z]);

          // ✅ Shape debug
          console.log("✅ Landmark count:", landmarkData.length);
          console.log("✅ First landmark:", landmarkData[0]);

          // ✅ Only send if shape is valid
          if (
            Array.isArray(landmarkData) &&
            landmarkData.length === 468 &&
            landmarkData.every((pt) => Array.isArray(pt) && pt.length === 3)
          ) {
            fetch("http://localhost:8000/predict", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                landmarks: landmarkData,
              }),
            })
              .then((res) => res.json())
              .then((data) => {
                console.log("📦 Full response from backend:", data);
                console.log("🎯 Predicted emotion:", data.predicted_emotion || data.emotion);
              })
              .catch((err) => console.error("❌ Prediction error:", err));
          } else {
            console.warn("❌ Landmark shape is invalid!");
          }
        } else {
          console.warn("🚫 No valid face landmarks detected.");
        }

        canvasCtx.restore();
      });

      const camera = new Camera(videoElement, {
        onFrame: async () => {
          await faceMesh.send({ image: videoElement });
        },
        width: 640,
        height: 480,
      });

      camera.start();
    }
  }, []);

  return (
    <div>
      <video ref={videoRef} style={{ display: "none" }}></video>
      <canvas ref={canvasRef} width="640" height="480" />
    </div>
  );
};

export default EmotionDetector;
