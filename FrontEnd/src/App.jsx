import './App.css';
import React from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';
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
import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

// GameWrapper Component with Emotion Detection
const GameWrapper = ({ children }) => {
  const location = useLocation();
  const [emotion, setEmotion] = React.useState(null);

  // Check if we're on a game route (excluding the dashboard)
  const isGameRoute = location.pathname.startsWith('/child/games/') && location.pathname !== '/child/games';

  // Handle the detected emotion
  const handleEmotionDetected = (detectedEmotion) => {
    try {
      setEmotion(detectedEmotion);
      fetch('http://localhost:5005/api/game/next-level', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ emotion: detectedEmotion }),
      })
        .then((res) => res.json())
        .then((data) => {
          console.log('🎯 Next Level Data:', data);
        })
        .catch((err) => {
          console.error('❌ Error calling next-level API:', err);
        });
    } catch (err) {
      console.error('❌ Error in handleEmotionDetected:', err);
    }
  };

  return (
    <>
      {/* Emotion detection for games */}
      {isGameRoute && (
        <BackgroundEmotionDetector isActive={true} onStopGame={handleEmotionDetected} />
      )}
      {React.cloneElement(children, { emotion })}
    </>
  );
};

// Emotion Detection Logic for Background
const BackgroundEmotionDetector = ({ isActive, onStopGame }) => {
  const videoRef = React.useRef(null);
  const capturedLandmarks = React.useRef(null);
  const intervalRef = React.useRef(null);
  const cameraRef = React.useRef(null);
  const lastWarningRef = React.useRef(0); // Throttle warnings

  React.useEffect(() => {
    if (typeof window !== 'undefined' && videoRef.current && isActive) {
      const videoElement = videoRef.current;

      try {
        const faceMesh = new FaceMesh({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
        });
        faceMesh.setOptions({
          maxNumFaces: 1,
          refineLandmarks: false, // Disable iris landmarks to ensure 468 points
          minDetectionConfidence: 0.5, // Increase for better detection
          minTrackingConfidence: 0.5, // Increase for better tracking
        });

        faceMesh.onResults((results) => {
          try {
            if (results.multiFaceLandmarks?.length > 0) {
              const landmarks = results.multiFaceLandmarks[0];
              const landmarkData = landmarks.map((pt) => [
                isFinite(pt.x) ? pt.x : 0,
                isFinite(pt.y) ? pt.y : 0,
                isFinite(pt.z) ? pt.z : 0,
              ]);

              // Validate landmark data
              if (landmarkData.length === 468 && landmarkData.every(pt => pt.length === 3 && pt.every(val => isFinite(val)))) {
                capturedLandmarks.current = landmarkData;
                console.log('Landmarks prepared:', landmarkData.slice(0, 5)); // Debug first 5 points
              } else {
                const now = Date.now();
                if (now - lastWarningRef.current > 5000) { // Throttle to every 5 seconds
                  console.warn('Invalid landmarks: incorrect length or invalid values', {
                    length: landmarkData.length,
                    sample: landmarkData.slice(0, 5),
                    hasInvalidValues: landmarkData.some(pt => !pt.every(val => isFinite(val))),
                  });
                  lastWarningRef.current = now;
                }
              }
            } else {
              const now = Date.now();
              if (now - lastWarningRef.current > 5000) { // Throttle to every 5 seconds
                console.warn('No face landmarks detected');
                lastWarningRef.current = now;
              }
            }
          } catch (err) {
            console.error('❌ Error processing FaceMesh results:', err);
          }
        });

        const camera = new Camera(videoElement, {
          onFrame: async () => {
            try {
              await faceMesh.send({ image: videoElement });
            } catch (err) {
              console.error('❌ Error sending frame to FaceMesh:', err);
            }
          },
          width: 640,
          height: 480,
        });

        cameraRef.current = camera;
        camera.start().catch((err) => {
          console.error('❌ Error starting camera:', err);
        });

<<<<<<< HEAD
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
                // onEmotionDetected(predictedEmotion);
                onStopGame(predictedEmotion);
              }
            })
            .catch((err) =>
              console.error(' Prediction error:', err)
            );
        }
      }, 5000);
=======
        intervalRef.current = setInterval(() => {
          if (capturedLandmarks.current) {
            console.log('Sending landmarks to backend:', capturedLandmarks.current.length); // Debug
            fetch('http://localhost:8000/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ landmarks: capturedLandmarks.current }),
            })
              .then((res) => res.json())
              .then((data) => {
                if (data.status === 'paused') return;
                const predictedEmotion = data.predicted_emotion || data.emotion;
                if (predictedEmotion) {
                  onStopGame(predictedEmotion);
                }
              })
              .catch((err) => console.error('❌ Prediction error:', err));
          } else {
            const now = Date.now();
            if (now - lastWarningRef.current > 5000) { // Throttle to every 5 seconds
              console.warn('No landmarks to send');
              lastWarningRef.current = now;
            }
          }
        }, 5000);
>>>>>>> 51717b948ab3b7d9569c179edaf8dc2911af9fb8

        return () => {
          clearInterval(intervalRef.current);
          if (cameraRef.current) cameraRef.current.stop();
        };
      } catch (err) {
        console.error('❌ Error initializing FaceMesh or camera:', err);
      }
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

// HomePage Component
const HomePage = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/SelectionPage');
  };

  return (
    <div className="App">
      <InteractiveElements />
      <div className="main-container">
        <h1 className="welcome-text">Welcome to Joyverse</h1>
        <p className="subtitle">Your Gateway to Digital Joy</p>
        <button className="get-started-btn"alcohol onClick={handleGetStarted}>
          Get Started
        </button>
      </div>
    </div>
  );
};

// Main App Component
function App() {
  return (
    <ChildProvider>
      <Router>
        <Routes>
          {/* Home and Selection Pages */}
          <Route path="/" element={<HomePage />} />
          <Route path="/SelectionPage" element={<SelectionPage />} />
          <Route path="/child" element={<ChildLogin />} />

          {/* Therapist Routes */}
          <Route path="/therapist" element={<TherapistLogin />} />
          <Route path="/child-info" element={<ChildInfoPage />} />
          <Route path="/childlist" element={<ChildList />} />

          {/* Game Routes */}
          <Route path="/child/games" element={<GameWrapper><GamesDashboard /></GameWrapper>} />
          <Route path="/child/games/word-wizard" element={<GameWrapper><WordWizard /></GameWrapper>} />
          <Route path="/child/games/hangman" element={<GameWrapper><Hangman /></GameWrapper>} />
          <Route path="/child/games/math-safari" element={<GameWrapper><MathSafari /></GameWrapper>} />
          <Route path="/child/games/memory" element={<GameWrapper><MemoryMatch /></GameWrapper>} />
          <Route path="/child/games/spelling" element={<GameWrapper><SpellingBee /></GameWrapper>} />
          <Route path="/child/games/science" element={<GameWrapper><ScienceQuest /></GameWrapper>} />
          <Route path="/child/games/puzzle" element={<GameWrapper><PuzzleWorld /></GameWrapper>} />
          <Route path="/child/games/reading" element={<GameWrapper><ReadingRace /></GameWrapper>} />
          <Route path="/child/games/art" element={<GameWrapper><ArtStudio /></GameWrapper>} />
          <Route path="/child/games/music" element={<GameWrapper><MusicMaker /></GameWrapper>} />
        </Routes>
      </Router>
    </ChildProvider>
  );
}

export default App;