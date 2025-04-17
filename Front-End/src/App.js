// joyverse/src/App.js
import './App.css';
import React from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { ChildProvider } from './context/ChildContext'; // Import the provider
import InteractiveElements from './components/InteractiveElements';
import SelectionPage from './components/SelectionPage';
import ChildLogin from './components/ChildLogin';
import TherapistLogin from './components/TherapistLogin';
import ChildInfoPage from './components/ChildInfoPage'; // Import the new component
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

function App() {
  return (
    <ChildProvider> {/* Wrap your Router with ChildProvider */}
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/select" element={<SelectionPage />} />
          <Route path="/child" element={<ChildLogin />} />
          <Route path="/child/games" element={<GamesDashboard />} />
          <Route path="/therapist" element={<TherapistLogin />} />
          <Route path="/child/games/word-wizard" element={<WordWizard />} />
          <Route path="/child/games/hangman" element={<Hangman />} />
          <Route path="/child/games/math-safari" element={<MathSafari />} />
          <Route path="/child/games/memory" element={<MemoryMatch />} />
          <Route path="/child/games/spelling" element={<SpellingBee />} />
          <Route path="/child/games/science" element={<ScienceQuest />} />
          <Route path="/child/games/puzzle" element={<PuzzleWorld />} />
          <Route path="/child/games/reading" element={<ReadingRace />} />
          <Route path="/child/games/art" element={<ArtStudio />} />
          <Route path="/child/games/music" element={<MusicMaker />} />
          <Route path="/child-info" element={<ChildInfoPage />} /> {/* New route for child's info */}
        </Routes>
      </Router>
    </ChildProvider>
  );
}

export default App;