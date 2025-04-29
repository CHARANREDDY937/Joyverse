import React, { useState, useRef, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import ChildLogin from './components/ChildLogin';
import TherapistLogin from './components/TherapistLogin';
import ChildInfoPage from './components/ChildInfoPage';
import GamesDashboard from './components/GamesDashboard';
import Hangman from './components/games/Hangman';
import WordWizard from './components/games/WordWizard';
import MathSafari from './components/games/MathSafari';
import MemoryMatch from './components/games/MemoryMatch';
import SpellingBee from './components/games/SpellingBee';
import ScienceQuest from './components/games/ScienceQuest';
import PuzzleWorld from './components/games/PuzzleWorld';
import ReadingRace from './components/games/ReadingRace';
import ArtStudio from './components/games/ArtStudio';
import MusicMaker from './components/games/MusicMaker';
import ChildList from './components/childlist';
import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';
import './App.css';

// -----------------------------
// BackgroundEmotionDetector Component
// -----------------------------
const BackgroundEmotionDetector = ({ isActive, onStopGame }) => {
  const videoRef = useRef(null);
  const capturedLandmarks = useRef(null);
  const intervalRef = useRef(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && videoRef.current && isActive) {
      const videoElement = videoRef.current;

      const faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.3,
        minTrackingConfidence: 0.3,
      });

      faceMesh.onResults((results) => {
        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
          const landmarks = results.multiFaceLandmarks[0];
          const landmarkData = landmarks.slice(0, 468).map((pt) => [pt.x, pt.y, pt.z]);

          if (
            Array.isArray(landmarkData) &&
            landmarkData.length === 468 &&
            landmarkData.every((pt) => Array.isArray(pt) && pt.length === 3)
          ) {
            capturedLandmarks.current = landmarkData;
          } else {
            console.warn('❌ Invalid landmark data structure');
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

      cameraRef.current = camera;
      camera.start();

      intervalRef.current = setInterval(() => {
        if (capturedLandmarks.current) {
          fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ landmarks: capturedLandmarks.current }),
          })
            .then((res) => res.json())
            .then((data) => {
              const predictedEmotion = data.predicted_emotion || data.emotion;
              console.log('🎯 Predicted emotion:', predictedEmotion);

              if (predictedEmotion) {
                onStopGame(predictedEmotion);
              }
            })
            .catch((err) => console.error('❌ Prediction error:', err));
        }
      }, 5000);

      return () => {
        clearInterval(intervalRef.current);
        if (cameraRef.current) {
          cameraRef.current.stop();
        }
      };
    } else if (cameraRef.current) {
      clearInterval(intervalRef.current);
      cameraRef.current.stop();
      cameraRef.current = null;
    }
  }, [isActive, onStopGame]);

  return (
    <video
      ref={videoRef}
      style={{ width: 0, height: 0, opacity: 0, position: 'absolute', zIndex: -1 }}
      playsInline
      muted
      autoPlay
    />
  );
};

// -----------------------------
// HomePage Component
// -----------------------------
const HomePage = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/child');   // ✅ Now navigating to /child instead of missing /select
  };

  return (
    <div className="App">
      <div className="main-container">
        <h1 className="welcome-text">Welcome to Joyverse</h1>
        <p className="subtitle">Your Gateway to Digital Joy</p>
        <button className="get-started-btn" onClick={handleGetStarted}>
          Get Started
        </button>
      </div>
    </div>
  );
};

// -----------------------------
// Handle Emotion Detection API
// -----------------------------
const handleEmotionDetected = (emotion) => {
  fetch('http://localhost:5005/api/game/next-level', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ emotion: emotion }),
  })
    .then((res) => res.json())
    .then((data) => {
      console.log('🎯 Next Level Data:', data);
    })
    .catch((err) => {
      console.error('❌ Error calling next-level API:', err);
    });
};

// -----------------------------
// GameWrapper Component
// -----------------------------
const GameWrapper = ({ children }) => {
  const location = useLocation();
  const isGameRoute = 
  location.pathname.startsWith('/child/games/science')||
  location.pathname.startsWith('/child/games/puzzle')||
  location.pathname.startsWith('/child/games/memory')||
  location.pathname.startsWith('/child/games/art')||
  location.pathname.startsWith('/child/games/music')||
  location.pathname.startsWith('/child/games/word-wizard')||
  location.pathname.startsWith('/child/games/math-safari')||
  location.pathname.startsWith('/child/games/spelling')||
  location.pathname.startsWith('/child/games/reading');

  return (
    <>
      {isGameRoute && <BackgroundEmotionDetector isActive={true} onStopGame={handleEmotionDetected} />}
      {children}
    </>
  );
};

// -----------------------------
// Main App Component
// -----------------------------
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/child" element={<ChildLogin />} />
        <Route
          path="/child/games"
          element={
            <GameWrapper>
              <GamesDashboard />
            </GameWrapper>
          }
        />
        <Route
          path="/child/games/word-wizard"
          element={
            <GameWrapper>
              <WordWizard />
            </GameWrapper>
          }
        />
        <Route
          path="/child/games/hangman"
          element={
            <GameWrapper>
              <Hangman />
            </GameWrapper>
          }
        />
        <Route
          path="/child/games/math-safari"
          element={
            <GameWrapper>
              <MathSafari />
            </GameWrapper>
          }
        />
        <Route
          path="/child/games/memory"
          element={
            <GameWrapper>
              <MemoryMatch />
            </GameWrapper>
          }
        />
        <Route
          path="/child/games/spelling"
          element={
            <GameWrapper>
              <SpellingBee />
            </GameWrapper>
          }
        />
        <Route
          path="/child/games/science"
          element={
            <GameWrapper>
              <ScienceQuest />
            </GameWrapper>
          }
        />
        <Route
          path="/child/games/puzzle"
          element={
            <GameWrapper>
              <PuzzleWorld />
            </GameWrapper>
          }
        />
        <Route
          path="/child/games/reading"
          element={
            <GameWrapper>
              <ReadingRace />
            </GameWrapper>
          }
        />
        <Route
          path="/child/games/art"
          element={
            <GameWrapper>
              <ArtStudio />
            </GameWrapper>
          }
        />
        <Route
          path="/child/games/music"
          element={
            <GameWrapper>
              <MusicMaker />
            </GameWrapper>
          }
        />
        <Route path="/therapist" element={<TherapistLogin />} />
        <Route path="/child-info" element={<ChildInfoPage />} />
        <Route path="/childlist" element={<ChildList />} />
      </Routes>
    </Router>
  );
}

export default App;