// import React, { useState, useRef, useEffect } from 'react';
// import {
//   BrowserRouter as Router,
//   Routes,
//   Route,
//   useNavigate,
//   useLocation
// } from 'react-router-dom';

// import InteractiveElements from './components/InteractiveElements';
// import SelectionPage from './components/SelectionPage';
// import ChildLogin from './components/ChildLogin';
// import TherapistLogin from './components/TherapistLogin';
// import ChildInfoPage from './components/ChildInfoPage';
// import GamesDashboard from './components/GamesDashboard';
// import Hangman from './components/games/Hangman';
// import WordWizard from './components/games/WordWizard';
// import MathSafari from './components/games/MathSafari';
// import MemoryMatch from './components/games/MemoryMatch';
// import SpellingBee from './components/games/SpellingBee';
// import ScienceQuest from './components/games/ScienceQuest';
// import PuzzleWorld from './components/games/PuzzleWorld';
// import ReadingRace from './components/games/ReadingRace';
// import ArtStudio from './components/games/ArtStudio';
// import MusicMaker from './components/games/MusicMaker';
// import ChildList from './components/childlist';
// import { FaceMesh } from '@mediapipe/face_mesh';
// import { Camera } from '@mediapipe/camera_utils';
// import './App.css';

// // -----------------------------
// // BackgroundEmotionDetector Component
// // -----------------------------
// const BackgroundEmotionDetector = ({ isActive, onStopGame }) => {
//   const videoRef = useRef(null);
//   const capturedLandmarks = useRef(null);
//   const intervalRef = useRef(null);
//   const cameraRef = useRef(null);

//   useEffect(() => {
//     if (typeof window !== 'undefined' && videoRef.current && isActive) {
//       console.log('Starting BackgroundEmotionDetector');
//       const videoElement = videoRef.current;

//       const faceMesh = new FaceMesh({
//         locateFile: (file) =>
//           `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
//       });
//       faceMesh.setOptions({
//         maxNumFaces: 1,
//         refineLandmarks: true,
//         minDetectionConfidence: 0.3,
//         minTrackingConfidence: 0.3,
//       });

//       faceMesh.onResults((results) => {
//         if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
//           const landmarks = results.multiFaceLandmarks[0];
//           const landmarkData = landmarks.slice(0, 468).map((pt) => [pt.x, pt.y, pt.z]);

//           if (
//             Array.isArray(landmarkData) &&
//             landmarkData.length === 468 &&
//             landmarkData.every((pt) => Array.isArray(pt) && pt.length === 3)
//           ) {
//             capturedLandmarks.current = landmarkData;
//           } else {
//             console.warn('❌ Invalid landmark data structure');
//           }
//         }
//       });

//       const camera = new Camera(videoElement, {
//         onFrame: async () => {
//           await faceMesh.send({ image: videoElement });
//         },
//         width: 640,
//         height: 480,
//       });

//       cameraRef.current = camera;
//       camera.start();

//       intervalRef.current = setInterval(() => {
//         if (capturedLandmarks.current) {
//           fetch('http://localhost:5000/api/predict-emotion', {
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({ landmarks: capturedLandmarks.current }),
//           })
//             .then((res) => res.json())
//             .then((data) => {
//               const predictedEmotion = data.emotion;
//               const themeUrl = data.theme;
      
//               console.log('🎯 Predicted emotion:', predictedEmotion);
//               console.log('🎨 Theme URL:', themeUrl);
      
//               // 🎯 Set game background if it's visible
//               const gameBg = document.getElementById('game-background');
//               if (gameBg && themeUrl) {
//                 gameBg.style.backgroundImage = `url(${themeUrl})`;
//                 // gameBg.style.backgroundSize = 'cover';
//                 // gameBg.style.backgroundPosition = 'center';
//                 // gameBg.style.transition = 'background-image 0.5s ease-in-out';
//               }
      
//               if (predictedEmotion) {
//                 // onEmotionDetected(predictedEmotion);
//                 onStopGame(predictedEmotion);
//               }
//             })
//             .catch((err) =>
//               console.error(' Prediction error:', err)
//             );
//         }
//       }, 5000);

//       return () => {
//         console.log('Stopping BackgroundEmotionDetector');
//         clearInterval(intervalRef.current);
//         if (cameraRef.current) {
//           cameraRef.current.stop();
//         }
//       };
//     } else if (cameraRef.current) {
//       console.log('Stopping BackgroundEmotionDetector (inactive)');
//       clearInterval(intervalRef.current);
//       cameraRef.current.stop();
//       cameraRef.current = null;
//     }
//   }, [isActive, onStopGame]);

//   return (
//     <video
//       ref={videoRef}
//       style={{ width: 0, height: 0, opacity: 0, position: 'absolute', zIndex: -1 }}
//       playsInline
//       muted
//       autoPlay
//     />
//   );
// };

// // -----------------------------
// // HomePage Component
// // -----------------------------
// const HomePage = () => {
//   const navigate = useNavigate();

//   const handleGetStarted = () => {
//     navigate('/SelectionPage');
//   };

//   return (
//     <div className="App">
//       <InteractiveElements />
//       <div className="main-container">
        
//         <h1 className="welcome-text">Welcome to Joyverse</h1>
//         <p className="subtitle">Your Gateway to Digital Joy</p>
//         <button className="get-started-btn" onClick={handleGetStarted}>
//           Get Started
//         </button>
//       </div>
//     </div>
//   );
// };

// // -----------------------------
// // GameWrapper Component
// // -----------------------------
// const GameWrapper = ({ children }) => {
//   const location = useLocation();
//   const [emotion, setEmotion] = useState(null);

//   // Only enable emotion detection for specific game routes, not the dashboard
//   const isGameRoute = location.pathname.startsWith('/child/games/') && location.pathname !== '/child/games';

//   const handleEmotionDetected = (emotion) => {
//     setEmotion(emotion);

//     fetch('http://localhost:5005/api/game/next-level', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({ emotion }),
//     })
//       .then((res) => res.json())
//       .then((data) => {
//         console.log('🎯 Next Level Data:', data);
//       })
//       .catch((err) => {
//         console.error('❌ Error calling next-level API:', err);
//       });
//   };

//   return (
//     <>
//       {isGameRoute && (
//         <BackgroundEmotionDetector isActive={true} onStopGame={handleEmotionDetected} />
//       )}
//       {React.cloneElement(children, { emotion })}
//     </>
//   );
// };

// // -----------------------------
// // Main App Component
// // -----------------------------
// function App() {
//   return (
//     <Router>
//       <Routes>
//         <Route path="/" element={<HomePage />} />
//         <Route path="/SelectionPage" element={<SelectionPage />} />
//         <Route path="/child" element={<ChildLogin />} />
//         <Route path="/therapist" element={<TherapistLogin />} />
//         <Route
//           path="/child/games"
//           element={
//             <GameWrapper>
//               <GamesDashboard />
//             </GameWrapper>
//           }
//         />
//         <Route
//           path="/child/games/word-wizard"
//           element={
//             <GameWrapper>
//               <WordWizard />
//             </GameWrapper>
//           }
//         />
//         <Route path="/child/games/hangman" element={<GameWrapper><Hangman /></GameWrapper>} />
//         <Route path="/child/games/math-safari" element={<GameWrapper><MathSafari /></GameWrapper>} />
//         <Route path="/child/games/memory" element={<GameWrapper><MemoryMatch /></GameWrapper>} />
//         <Route path="/child/games/spelling" element={<GameWrapper><SpellingBee /></GameWrapper>} />
//         <Route path="/child/games/science" element={<GameWrapper><ScienceQuest /></GameWrapper>} />
//         <Route path="/child/games/puzzle" element={<GameWrapper><PuzzleWorld /></GameWrapper>} />
//         <Route path="/child/games/reading" element={<GameWrapper><ReadingRace /></GameWrapper>} />
//         <Route path="/child/games/art" element={<GameWrapper><ArtStudio /></GameWrapper>} />
//         <Route path="/child/games/music" element={<GameWrapper><MusicMaker /></GameWrapper>} />
//         <Route path="/child-info" element={<ChildInfoPage />} />
//         <Route path="/childlist" element={<ChildList />} />
//       </Routes>
//     </Router>
//   );
// }

// export default App;
// Add this import at the top with other imports
// import ChildProgress from './components/ChildProgress';

// import React, { useState, useRef, useEffect } from 'react';
// import {
//   BrowserRouter as Router,
//   Routes,
//   Route,
//   useNavigate,
//   useLocation,
// } from 'react-router-dom';

// import InteractiveElements from './components/InteractiveElements';
// import SelectionPage from './components/SelectionPage';
// import ChildLogin from './components/ChildLogin';
// import TherapistLogin from './components/TherapistLogin';
// import ChildInfoPage from './components/ChildInfoPage';
// import GamesDashboard from './components/GamesDashboard';
// import Hangman from './components/games/Hangman';
// import WordWizard from './components/games/WordWizard';
// import MathSafari from './components/games/MathSafari';
// import MemoryMatch from './components/games/MemoryMatch';
// import SpellingBee from './components/games/SpellingBee';
// import ScienceQuest from './components/games/ScienceQuest';
// import PuzzleWorld from './components/games/PuzzleWorld';
// import ReadingRace from './components/games/ReadingRace';
// import ArtStudio from './components/games/ArtStudio';
// import MusicMaker from './components/games/MusicMaker';
// import ChildList from './components/childlist';

// import { FaceMesh } from '@mediapipe/face_mesh';
// import { Camera } from '@mediapipe/camera_utils';
// import { ChildProvider } from './context/ChildContext';

// import './App.css';

// // -----------------------------
// // BackgroundEmotionDetector
// // -----------------------------
// const BackgroundEmotionDetector = ({ isActive, onEmotionDetected }) => {
//   const videoRef = useRef(null);
//   const capturedLandmarks = useRef(null);
//   const intervalRef = useRef(null);
//   const cameraRef = useRef(null);

//   useEffect(() => {
//     if (!isActive || !videoRef.current) return;

//     console.log('🎥 Starting BackgroundEmotionDetector');
//     const videoElement = videoRef.current;

//     const faceMesh = new FaceMesh({
//       locateFile: (file) =>
//         `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
//     });

//     faceMesh.setOptions({
//       maxNumFaces: 1,
//       refineLandmarks: true,
//       minDetectionConfidence: 0.3,
//       minTrackingConfidence: 0.3,
//     });

//     faceMesh.onResults((results) => {
//       if (results.multiFaceLandmarks?.length) {
//         const landmarks = results.multiFaceLandmarks[0];
//         const landmarkData = landmarks.slice(0, 468).map((pt) => [pt.x, pt.y, pt.z]);

//         if (
//           Array.isArray(landmarkData) &&
//           landmarkData.length === 468 &&
//           landmarkData.every((pt) => Array.isArray(pt) && pt.length === 3)
//         ) {
//           capturedLandmarks.current = landmarkData;
//         } else {
//           console.warn('❌ Invalid landmark data structure');
//         }
//       }
//     });

//     const camera = new Camera(videoElement, {
//       onFrame: async () => {
//         await faceMesh.send({ image: videoElement });
//       },
//       width: 640,
//       height: 480,
//     });

//     cameraRef.current = camera;
//     camera.start();

//     intervalRef.current = setInterval(() => {
//       if (capturedLandmarks.current) {
//         fetch('http://localhost:5000/api/predict-emotion', {
//           method: 'POST',
//           headers: { 'Content-Type': 'application/json' },
//           body: JSON.stringify({ landmarks: capturedLandmarks.current }),
//         })
//           .then((res) => res.json())
//           .then((data) => {
//             const { emotion, theme } = data;
//             console.log('🎯 Emotion:', emotion);
//             console.log('🎨 Theme:', theme);

//             const gameBg = document.getElementById('game-background');
//             if (gameBg && theme) {
//               gameBg.style.backgroundImage = `url(${theme})`;
//             }

//             if (emotion) {
//               onEmotionDetected(emotion);
//             }
//           })
//           .catch((err) => {
//             console.error('❌ Prediction error:', err);
//           });
//       }
//     }, 5000);

//     return () => {
//       console.log('🛑 Cleaning up BackgroundEmotionDetector');
//       clearInterval(intervalRef.current);
//       if (cameraRef.current) {
//         cameraRef.current.stop();
//         cameraRef.current = null;
//       }
//     };
//   }, [isActive, onEmotionDetected]);

//   return (
//     <video
//       ref={videoRef}
//       style={{ width: 0, height: 0, opacity: 0, position: 'absolute', zIndex: -1 }}
//       playsInline
//       muted
//       autoPlay
//     />
//   );
// };

// // -----------------------------
// // HomePage Component
// // -----------------------------
// const HomePage = () => {
//   const navigate = useNavigate();

//   return (
//     <div className="App">
//       <InteractiveElements />
//       <div className="main-container">
//         <h1 className="welcome-text">Welcome to Joyverse</h1>
//         <p className="subtitle">Your Gateway to Digital Joy</p>
//         <button className="get-started-btn" onClick={() => navigate('/SelectionPage')}>
//           Get Started
//         </button>
//       </div>
//     </div>
//   );
// };

// // -----------------------------
// // GameWrapper Component
// // -----------------------------
// const GameWrapper = ({ children }) => {
//   const location = useLocation();
//   const [emotion, setEmotion] = useState(null);

//   const isGameRoute =
//     location.pathname.startsWith('/child/games/') &&
//     location.pathname !== '/child/games';

//   const handleEmotionDetected = (emotion) => {
//     setEmotion(emotion);

//     fetch('http://localhost:5005/api/game/next-level', {
//       method: 'POST',
//       headers: { 'Content-Type': 'application/json' },
//       body: JSON.stringify({ emotion }),
//     })
//       .then((res) => res.json())
//       .then((data) => {
//         console.log('🎯 Next Level Data:', data);
//       })
//       .catch((err) => {
//         console.error('❌ Error calling next-level API:', err);
//       });
//   };

//   return (
//     <>
//       {isGameRoute && (
//         <BackgroundEmotionDetector
//           isActive={true}
//           onEmotionDetected={handleEmotionDetected}
//         />
//       )}
//       {React.isValidElement(children)
//         ? React.cloneElement(children, { emotion })
//         : children}
//     </>
//   );
// };

// // -----------------------------
// // Main App Component
// // -----------------------------
// function App() {
//   return (
//     <ChildProvider>
//       <Router>
//         <Routes>
//           <Route path="/" element={<HomePage />} />
//           <Route path="/SelectionPage" element={<SelectionPage />} />
//           <Route path="/child" element={<ChildLogin />} />
//           <Route path="/therapist" element={<TherapistLogin />} />
//           <Route path="/child-info" element={<ChildInfoPage />} />
//           <Route path="/childlist" element={<ChildList />} />

//           {/* Game Dashboard */}
//           <Route
//             path="/child/games"
//             element={<GameWrapper><GamesDashboard /></GameWrapper>}
//           />

//           {/* Game Routes */}
//           <Route
//             path="/child/games/word-wizard"
//             element={<GameWrapper><WordWizard /></GameWrapper>}
//           />
//           <Route
//             path="/child/games/hangman"
//             element={<GameWrapper><Hangman /></GameWrapper>}
//           />
//           <Route
//             path="/child/games/math-safari"
//             element={<GameWrapper><MathSafari /></GameWrapper>}
//           />
//           <Route
//             path="/child/games/memory"
//             element={<GameWrapper><MemoryMatch /></GameWrapper>}
//           />
//           <Route
//             path="/child/games/spelling"
//             element={<GameWrapper><SpellingBee /></GameWrapper>}
//           />
//           <Route
//             path="/child/games/science"
//             element={<GameWrapper><ScienceQuest /></GameWrapper>}
//           />
//           <Route
//             path="/child/games/puzzle"
//             element={<GameWrapper><PuzzleWorld /></GameWrapper>}
//           />
//           <Route
//             path="/child/games/reading"
//             element={<GameWrapper><ReadingRace /></GameWrapper>}
//           />
//           <Route
//             path="/child/games/art"
//             element={<GameWrapper><ArtStudio /></GameWrapper>}
//           />
//           <Route
//             path="/child/games/music"
//             element={<GameWrapper><MusicMaker /></GameWrapper>}
//           />
//           <Route path="/child-progress/:username" element={<ChildProgress />} />
//         </Routes>
//       </Router>
//     </ChildProvider>
//   );
// }

// export default App;
import ChildProgress from './components/ChildProgress';
import React, { useState, useRef, useEffect } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
  useLocation,
} from 'react-router-dom';

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
import { ChildProvider } from './context/ChildContext';
import './App.css';

// Define API base URLs based on environment
const isProduction = process.env.NODE_ENV === 'production';
const FASTAPI_BASE_URL = isProduction
  ? 'https://api-pmbi.onrender.com'
  : 'http://localhost:8000'; // Adjust to your local FastAPI port
const NODE_BASE_URL = isProduction
  ? 'https://backend-brmn.onrender.com'
  : 'http://localhost:5000'; // Local Node.js port

// -----------------------------
// BackgroundEmotionDetector
// -----------------------------
const BackgroundEmotionDetector = ({ isActive, onEmotionDetected }) => {
  const videoRef = useRef(null);
  const capturedLandmarks = useRef(null);
  const intervalRef = useRef(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    if (!isActive || !videoRef.current) return;

    console.log('🎥 Starting BackgroundEmotionDetector');
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
      if (results.multiFaceLandmarks?.length) {
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
        fetch(`${NODE_BASE_URL}/api/predict-emotion`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ landmarks: capturedLandmarks.current }),
        })
          .then((res) => res.json())
          .then((data) => {
            const { emotion, theme } = data;
            console.log('🎯 Emotion:', emotion);
            console.log('🎨 Theme:', theme);

            const gameBg = document.getElementById('game-background');
            if (gameBg && theme) {
              gameBg.style.backgroundImage = `url(${theme})`;
            }

            if (emotion) {
              onEmotionDetected(emotion);
            }
          })
          .catch((err) => {
            console.error('❌ Prediction error:', err);
          });
      }
    }, 5000);

    return () => {
      console.log('🛑 Cleaning up BackgroundEmotionDetector');
      clearInterval(intervalRef.current);
      if (cameraRef.current) {
        cameraRef.current.stop();
        cameraRef.current = null;
      }
    };
  }, [isActive, onEmotionDetected]);

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

// -----------------------------
// GameWrapper Component
// -----------------------------
const GameWrapper = ({ children }) => {
  const location = useLocation();
  const [emotion, setEmotion] = useState(null);

  const isGameRoute =
    location.pathname.startsWith('/child/games/') &&
    location.pathname !== '/child/games';

  const handleEmotionDetected = (emotion) => {
    setEmotion(emotion);

    fetch(`${FASTAPI_BASE_URL}/api/game/next-level`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ emotion }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log('🎯 Next Level Data:', data);
      })
      .catch((err) => {
        console.error('❌ Error calling next-level API:', err);
      });
  };

  return (
    <>
      {isGameRoute && (
        <BackgroundEmotionDetector
          isActive={true}
          onEmotionDetected={handleEmotionDetected}
        />
      )}
      {React.isValidElement(children)
        ? React.cloneElement(children, { emotion })
        : children}
    </>
  );
};

// -----------------------------
// Main App Component
// -----------------------------
function App() {
  return (
    <ChildProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/SelectionPage" element={<SelectionPage />} />
          <Route path="/child" element={<ChildLogin />} />
          <Route path="/therapist" element={<TherapistLogin />} />
          <Route path="/child-info" element={<ChildInfoPage />} />
          <Route path="/childlist" element={<ChildList />} />
          <Route
            path="/child/games"
            element={<GameWrapper><GamesDashboard /></GameWrapper>}
          />
          <Route
            path="/child/games/word-wizard"
            element={<GameWrapper><WordWizard /></GameWrapper>}
          />
          <Route
            path="/child/games/hangman"
            element={<GameWrapper><Hangman /></GameWrapper>}
          />
          <Route
            path="/child/games/math-safari"
            element={<GameWrapper><MathSafari /></GameWrapper>}
          />
          <Route
            path="/child/games/memory"
            element={<GameWrapper><MemoryMatch /></GameWrapper>}
          />
          <Route
            path="/child/games/spelling"
            element={<GameWrapper><SpellingBee /></GameWrapper>}
          />
          <Route
            path="/child/games/science"
            element={<GameWrapper><ScienceQuest /></GameWrapper>}
          />
          <Route
            path="/child/games/puzzle"
            element={<GameWrapper><PuzzleWorld /></GameWrapper>}
          />
          <Route
            path="/child/games/reading"
            element={<GameWrapper><ReadingRace /></GameWrapper>}
          />
          <Route
            path="/child/games/art"
            element={<GameWrapper><ArtStudio /></GameWrapper>}
          />
          <Route
            path="/child/games/music"
            element={<GameWrapper><MusicMaker /></GameWrapper>}
          />
          <Route path="/child-progress/:username" element={<ChildProgress />} />
        </Routes>
      </Router>
    </ChildProvider>
  );
}

export default App;