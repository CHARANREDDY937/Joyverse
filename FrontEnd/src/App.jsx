import './App.css';
import React, { useRef, useEffect } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
  useLocation,
} from 'react-router-dom';
import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

import { ChildProvider } from './context/ChildContext';
import InteractiveElements from './components/InteractiveElements';
import SelectionPage from './components/SelectionPage';
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

const BackgroundEmotionDetector = ({ isActive, onEmotionDetected }) => {
  const videoRef = useRef(null);
  const capturedLandmarks = useRef(null);
  const intervalRef = useRef(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && videoRef.current && isActive) {
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
          const landmarkData = landmarks.slice(0, 468).map((pt) => [
            pt.x,
            pt.y,
            pt.z,
          ]);

          if (
            Array.isArray(landmarkData) &&
            landmarkData.length === 468 &&
            landmarkData.every(
              (pt) => Array.isArray(pt) && pt.length === 3
            )
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
          fetch('http://localhost:5000/api/predict-emotion', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ landmarks: capturedLandmarks.current }),
          })
            .then((res) => res.json())
            .then((data) => {
              const predictedEmotion = data.emotion;
              const themeUrl = data.theme;
      
              console.log('🎯 Predicted emotion:', predictedEmotion);
              console.log('🎨 Theme URL:', themeUrl);
      
              // 🎯 Set game background if it's visible
              const gameBg = document.getElementById('game-background');
              if (gameBg && themeUrl) {
                gameBg.style.backgroundImage = `url(${themeUrl})`;
                // gameBg.style.backgroundSize = 'cover';
                // gameBg.style.backgroundPosition = 'center';
                // gameBg.style.transition = 'background-image 0.5s ease-in-out';
              }
      
              if (predictedEmotion) {
                onEmotionDetected(predictedEmotion);
              }
            })
            .catch((err) =>
              console.error('❌ Prediction error:', err)
            );
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
  }, [isActive, onEmotionDetected]);

  return (
    <video
      ref={videoRef}
      style={{ width: 0, height: 0, opacity: 0, position: 'absolute' }}
      playsInline
      muted
      autoPlay
    />
  );
};

const HomePage = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/select');
  };

  return (
    <div className="App">
      <InteractiveElements />
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

const handleEmotionDetected = (emotion) => {
  // Use the emotion however your app logic needs it
  console.log('📥 Emotion received in handler:', emotion);

  // Optionally send to backend or trigger game adaptation logic
  fetch('http://localhost:5005/api/game/next-level', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ emotion: emotion }),
  })
    .then((res) => res.json())
    .then((data) => {
      console.log('🎮 Next Level Data:', data);
    })
    .catch((err) => {
      console.error('❌ Error calling next-level API:', err);
    });
};

const GameWrapperContent = ({ children }) => {
  const location = useLocation();
  const isGameRoute =
    location.pathname.includes('/child/games/') &&
    location.pathname !== '/child/games';

  return (
    <>
      {isGameRoute && (
        <BackgroundEmotionDetector
          isActive={true}
          onEmotionDetected={handleEmotionDetected}
        />
      )}
      {children}
    </>
  );
};

const GameWrapper = ({ children }) => {
  return <GameWrapperContent>{children}</GameWrapperContent>;
};

function App() {
  return (
    <Router>
      <ChildProvider>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/select" element={<SelectionPage />} />
          <Route path="/child" element={<ChildLogin />} />
          <Route path="/therapist" element={<TherapistLogin />} />
          <Route path="/child-info" element={<ChildInfoPage />} />
          <Route path="/childlist" element={<ChildList />} />

          <Route path="/child/games" element={<GamesDashboard />} />
          <Route
            path="/child/games/*"
            element={
              <GameWrapper>
                <Routes>
                  <Route path="word-wizard" element={<WordWizard />} />
                  <Route path="hangman" element={<Hangman />} />
                  <Route path="math-safari" element={<MathSafari />} />
                  <Route path="memory-match" element={<MemoryMatch />} />
                  <Route path="spelling-bee" element={<SpellingBee />} />
                  <Route path="science-quest" element={<ScienceQuest />} />
                  <Route path="puzzle-world" element={<PuzzleWorld />} />
                  <Route path="reading-race" element={<ReadingRace />} />
                  <Route path="art-studio" element={<ArtStudio />} />
                  <Route path="music-maker" element={<MusicMaker />} />
                </Routes>
              </GameWrapper>
            }
          />
        </Routes>
      </ChildProvider>
    </Router>
  );
}

export default App;
