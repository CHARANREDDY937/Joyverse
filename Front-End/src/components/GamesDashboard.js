import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './GamesDashboard.css';

const GamesDashboard = () => {
  const navigate = useNavigate();

  const handleGameClick = () => {
    navigate('/games/Hangman');
  };

  const cardVariants = {
    initial: { scale: 0.8, opacity: 0 },
    animate: { 
      scale: 1, 
      opacity: 1,
      transition: { duration: 0.5 }
    },
    hover: { 
      scale: 1.05,
      y: -10,
      transition: { duration: 0.3 }
    }
  };

  return (
    <div className="games-dashboard">
      <div className="dashboard-header">
        <h1 className="game-title">
          <span className="hang-text">Mystery </span>
          <span className="man-text">Words</span>
          <span className="character">😊</span>
        </h1>
        <p>Guess the hidden word one letter at a time!</p>
      </div>

      <motion.div 
        className="game-card-container"
        initial="initial"
        animate="animate"
        whileHover="hover"
        variants={cardVariants}
      >
        <div className="game-card" onClick={handleGameClick}>
          <div className="game-icon">🎯</div>
          <h2>Word Adventure</h2>
          <p>Join an exciting word-guessing journey! Choose from different categories, guess letters, and save your character before the chances run out.</p>
          <div className="game-features">
            <span>🎲 Choose Categories</span>
            <span>💭 Guess Letters</span>
            <span>⭐ Win Points</span>
          </div>
          <button className="start-adventure-btn">
            Play Now
          </button>
        </div>
      </motion.div>
    </div>
  );
};

export default GamesDashboard; 