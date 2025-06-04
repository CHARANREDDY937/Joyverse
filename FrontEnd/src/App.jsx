import React, { useState, useEffect, useRef, useContext } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';

// Individual Component Imports
import ChildProgress from './components/ChildProgress';
import InteractiveElements from './components/InteractiveElements';
import SelectionPage from './components/SelectionPage';
import ChildLogin from './components/ChildLogin';
import TherapistLogin from './components/TherapistLogin';
import AdminLogin from './components/AdminLogin';
import AdminDashboard from './components/AdminDashboard';
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

import { ChildProvider, ChildContext } from './context/ChildContext';
import './App.css';

const isProduction = process.env.NODE_ENV === 'production';
const FASTAPI_BASE_URL = isProduction ? 'https://api-pmbi.onrender.com' : 'http://localhost:8000';
const NODE_BASE_URL = isProduction ? 'https://backend-brmn.onrender.com' : 'http://localhost:5000';

const gameRouteToName = {
  '/child/games/word-wizard': 'WordWizard',
  '/child/games/hangman': 'Hangman',
  '/child/games/math-safari': 'MathSafari',
  '/child/games/memory': 'MemoryMatch',
  '/child/games/spelling': 'SpellingBee',
  '/child/games/science': 'ScienceQuest',
  '/child/games/puzzle': 'PuzzleWorld',
  '/child/games/reading': 'ReadingRace',
  '/child/games/art': 'ArtStudio',
  '/child/games/music': 'MusicMaker',
};
const BackgroundEmotionDetector = ({ isActive, onEmotionDetected }) => {
  const videoRef = useRef(null);
  const capturedLandmarks = useRef(null);
  const intervalRef = useRef(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    if (!isActive || !videoRef.current) return;

    const loadMediaPipe = async () => {
      const { FaceMesh } = await import('@mediapipe/face_mesh');
      const { Camera } = await import('@mediapipe/camera_utils');
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
        const landmarks = results.multiFaceLandmarks?.[0];
        if (landmarks) {
          const landmarkData = landmarks.slice(0, 468).map(pt => [pt.x, pt.y, pt.z]);
          capturedLandmarks.current = landmarkData;
        }
      });

      const camera = new Camera(videoRef.current, {
        onFrame: async () => await faceMesh.send({ image: videoRef.current }),
        width: 640,
        height: 480,
      });

      cameraRef.current = camera;
      camera.start();

      intervalRef.current = setInterval(() => {
        if (!capturedLandmarks.current) return;

        fetch(`${NODE_BASE_URL}/api/predict-emotion`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ landmarks: capturedLandmarks.current }),
        })
          .then(res => res.json())
          .then(({ emotion, theme }) => {
            if (theme) {
              const bg = document.getElementById('game-background');
              if (bg) bg.style.backgroundImage = `url(${theme})`;
            }
            if (emotion) onEmotionDetected(emotion);
          })
          .catch(console.error);
      }, 5000);
    };

    loadMediaPipe();

    return () => {
      clearInterval(intervalRef.current);
      cameraRef.current?.stop();
    };
  }, [isActive, onEmotionDetected]);

  return <video ref={videoRef} style={{ display: 'none' }} autoPlay muted playsInline />;
};

const HomePage = () => {
  const navigate = useNavigate();
  return (
    <div className="App">
      <InteractiveElements />
      <div className="main-container">
        <h1 className="welcome-text">Welcome to Joyverse</h1>
        <p className="subtitle">Your Gateway to Digital Joy</p>
        <button className="get-started-btn" onClick={() => navigate('/SelectionPage')}>
          Get Started
        </button>
      </div>
    </div>
  );
};

const GameWrapper = ({ children }) => {
  const location = useLocation();
  const [emotion, setEmotion] = useState(null);
  const { user } = useContext(ChildContext);
  const previousPath = useRef(null);

  const isGame = location.pathname.startsWith('/child/games/') && location.pathname !== '/child/games';

  const handleEmotionDetected = (emotion) => {
    setEmotion(emotion);
    fetch(`${FASTAPI_BASE_URL}/api/game/next-level`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ emotion }),
    }).catch(console.error);
  };

  useEffect(() => {
    const prev = previousPath.current;
    const current = location.pathname;
    if (current === '/child/games' && prev?.startsWith('/child/games/') && user?.username) {
      const game = gameRouteToName[prev] || 'UnknownGame';
      fetch(`${NODE_BASE_URL}/api/emotion-summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user.username, game }),
      }).catch(console.error);
    }
    previousPath.current = current;
  }, [location.pathname, user]);

  return (
    <>
      {isGame && <BackgroundEmotionDetector isActive onEmotionDetected={handleEmotionDetected} />}
      {React.isValidElement(children) ? React.cloneElement(children, { emotion }) : children}
    </>
  );
};

const ProtectedAdminRoute = ({ children }) => {
  const navigate = useNavigate();
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('adminToken');
    if (!token) navigate('/admin/login');
    else setIsAuthenticated(true);
    setLoading(false);
  }, [navigate]);

  if (loading) return <div>Loading...</div>;
  return isAuthenticated ? children : null;
};

export default function App() {
  return (
    <ChildProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/SelectionPage" element={<SelectionPage />} />
          <Route path="/admin/login" element={<AdminLogin />} />
          <Route path="/admin/dashboard" element={<ProtectedAdminRoute><AdminDashboard /></ProtectedAdminRoute>} />
          <Route path="/child" element={<ChildLogin />} />
          <Route path="/therapist" element={<TherapistLogin />} />
          <Route path="/child-info" element={<ChildInfoPage />} />
          <Route path="/childlist" element={<ChildList />} />
          <Route path="/child-progress/:username" element={<ChildProgress />} />

          <Route path="/child/games" element={<GameWrapper><GamesDashboard /></GameWrapper>} />
          <Route path="/child/games/word-wizard" element={<GameWrapper><WordWizard /></GameWrapper>} />
          {/* <Route path="/child/games/hangman" element={<GameWrapper><Hangman /></GameWrapper>} /> */}
          <Route path="/child/games/math-safari" element={<GameWrapper><MathSafari /></GameWrapper>} />
          <Route path="/child/games/memory" element={<GameWrapper><MemoryMatch /></GameWrapper>} />
          <Route path="/child/games/spelling" element={<GameWrapper><SpellingBee /></GameWrapper>} />
          {/* <Route path="/child/games/science" element={<GameWrapper><ScienceQuest /></GameWrapper>} /> */}
          <Route path="/child/games/puzzle" element={<GameWrapper><PuzzleWorld /></GameWrapper>} />
          {/* <Route path="/child/games/reading" element={<GameWrapper><ReadingRace /></GameWrapper>} /> */}
          <Route path="/child/games/art" element={<GameWrapper><ArtStudio /></GameWrapper>} />
          {/* <Route path="/child/games/music" element={<GameWrapper><MusicMaker /></GameWrapper>} /> */}
        </Routes>
      </Router>
    </ChildProvider>
  );
}
