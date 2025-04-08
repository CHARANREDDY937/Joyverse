import './App.css';
import React from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import InteractiveElements from './components/InteractiveElements';
import SelectionPage from './components/SelectionPage';
import ChildLogin from './components/ChildLogin';
import TherapistLogin from './components/TherapistLogin';
import GamesDashboard from './components/GamesDashboard';
import Hangman from './components/games/Hangman';

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
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/select" element={<SelectionPage />} />
        <Route path="/child" element={<ChildLogin />} />
        <Route path="/child/games" element={<GamesDashboard />} />
        <Route path="/therapist" element={<TherapistLogin />} />
        <Route path="/games/hangman" element={<Hangman />} />
      </Routes>
    </Router>
  );
}

export default App;
