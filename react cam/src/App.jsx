import React, { useRef, useEffect } from "react";
import { FaceMesh } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";

const BackgroundEmotionDetector = () => {
  const videoRef = useRef(null);
  const capturedLandmarks = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (typeof window !== "undefined" && videoRef.current) {
      const videoElement = videoRef.current;

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
        if (
          results.multiFaceLandmarks &&
          results.multiFaceLandmarks.length > 0
        ) {
          const landmarks = results.multiFaceLandmarks[0];
          const landmarkData = landmarks.slice(0, 468).map((pt) => [pt.x, pt.y, pt.z]);

          if (
            Array.isArray(landmarkData) &&
            landmarkData.length === 468 &&
            landmarkData.every((pt) => Array.isArray(pt) && pt.length === 3)
          ) {
            capturedLandmarks.current = landmarkData;
          } else {
            console.warn("❌ Invalid landmark data structure");
          }
        }
      });

      const camera = new Camera(videoElement, {
        onFrame: async () => {
          await faceMesh.send({ image: videoElement });
        },
        width: 640,
        height: 480,
      });

      camera.start();

      // Send emotion data every 5 seconds
      intervalRef.current = setInterval(() => {
        if (capturedLandmarks.current) {
          fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ landmarks: capturedLandmarks.current }),
          })
            .then((res) => res.json())
            .then((data) => {
              console.log("🎯 Predicted emotion:", data.predicted_emotion || data.emotion);
            })
            .catch((err) => console.error("❌ Prediction error:", err));
        }
      }, 5000);

      return () => {
        clearInterval(intervalRef.current);
        camera.stop();
      };
    }
  }, []);

  return (
    <>
      <video
        ref={videoRef}
        style={{ width: 0, height: 0, opacity: 0, position: 'absolute', zIndex: -1 }}
        playsInline
        muted
        autoPlay
      />
    </>
  );
};

export default BackgroundEmotionDetector;
