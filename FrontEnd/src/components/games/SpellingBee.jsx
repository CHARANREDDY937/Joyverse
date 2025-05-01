import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './SpellingBee.css';

const WORD_LISTS = {
  easy: [
    { word: 'cat', hint: '🐱 A furry pet that meows' },
    { word: 'dog', hint: '🐕 A loyal furry friend' },
    { word: 'sun', hint: '☀️ Shines in the sky' },
    { word: 'hat', hint: '🎩 Wear it on your head' },
    { word: 'bee', hint: '🐝 Makes honey' }
  ],
  medium: [
    { word: 'apple', hint: '🍎 A sweet red fruit' },
    { word: 'beach', hint: '🏖️ Sandy shore by the ocean' },
    { word: 'cloud', hint: '☁️ Float in the sky' },
    { word: 'tiger', hint: '🐯 Striped big cat' },
    { word: 'house', hint: '🏠 Place to live' }
  ],
  hard: [
    { word: 'elephant', hint: '🐘 Large gray animal with a trunk' },
    { word: 'butterfly', hint: '🦋 Colorful flying insect' },
    { word: 'rainbow', hint: '🌈 Colorful arc in the sky' },
    { word: 'penguin', hint: '🐧 Bird that swims but cannot fly' },
    { word: 'dolphin', hint: '🐬 Smart sea mammal' }
  ]
};

const SpellingBee = () => {
  const navigate = useNavigate();
  const [level, setLevel] = useState('easy');
  const [currentWord, setCurrentWord] = useState(null);
  const [userInput, setUserInput] = useState('');
  const [score, setScore] = useState(0);
  const [streak, setStreak] = useState(0);
  const [showHint, setShowHint] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [error, setError] = useState(null);

  const selectNewWord = useCallback(() => {
    const words = WORD_LISTS[level];
    const randomWord = words[Math.floor(Math.random() * words.length)];
    setCurrentWord(randomWord);
    setUserInput('');
    setShowHint(false);
    setFeedback(null);
  }, [level]);

  useEffect(() => {
    selectNewWord();
  }, [level, selectNewWord]);

  const speakWord = () => {
    const utterance = new SpeechSynthesisUtterance(currentWord.word);
    utterance.rate = 0.8;
    window.speechSynthesis.speak(utterance);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (userInput.toLowerCase() === currentWord.word.toLowerCase()) {
      setFeedback({ type: 'success', message: 'Correct! 🎉' });
      setScore(prev => prev + (streak + 1) * 10);
      setStreak(prev => prev + 1);
      
      setTimeout(() => {
        selectNewWord();
      }, 1500);
    } else {
      setFeedback({ type: 'error', message: 'Try again! 💪' });
      setStreak(0);
    }
  };

  const resetGame = async () => {
    try {
      const response = await fetch('http://localhost:8000/clear_emotions_log', {
        method: 'POST',
      });
      const data = await response.json();
      if (data.status !== 'success') {
        setError('Failed to clear emotions log: ' + data.message);
      } else {
        setError(null);
      }
    } catch (error) {
      setError('Error clearing emotions log: ' + error.message);
    }

    setScore(0);
    setStreak(0);
    setLevel('easy');
    setUserInput('');
    setShowHint(false);
    setFeedback(null);
    selectNewWord();
  };

  // Initialize game and clear emotions log on mount
  useEffect(() => {
    const initializeGame = async () => {
      try {
        const response = await fetch('http://localhost:8000/clear_emotions_log', {
          method: 'POST',
        });
        const data = await response.json();
        if (data.status !== 'success') {
          setError('Failed to clear emotions log: ' + data.message);
        } else {
          setError(null);
        }
      } catch (error) {
        setError('Error clearing emotions log: ' + error.message);
      }
    };

    initializeGame();

    // Cleanup: Signal external emotion detection system to stop logging (if applicable)
    return () => {
      // Note: Add logic here to stop external emotion detection (e.g., webcam) if controlled by the app.
      // Currently, we assume the external system stops when the game unmounts.
      console.log('SpellingBee unmounted: Emotion logging should stop.');
    };
  }, []);

  return (
    <div className="spelling-bee">
      <motion.button
        className="back-button"
        onClick={() => navigate('/child/games')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Games
      </motion.button>

      <div className="game-header">
        <h1>Spelling Bee</h1>
        <div className="game-info">
          <span className="score">Score: {score}</span>
          <span className="streak">Streak: {streak} 🔥</span>
          <motion.button
            className="reset-btn"
            onClick={resetGame}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Reset
          </motion.button>
        </div>
      </div>

      {error && (
        <motion.p
          className="error-message"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {error}
        </motion.p>
      )}

      <div className="level-selector">
        {Object.keys(WORD_LISTS).map((lvl) => (
          <motion.button
            key={lvl}
            className={`level-btn ${level === lvl ? 'active' : ''}`}
            onClick={() => setLevel(lvl)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {lvl.charAt(0).toUpperCase() + lvl.slice(1)}
          </motion.button>
        ))}
      </div>

      {currentWord && (
        <motion.div 
          className="game-container"
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
        >
          <div className="word-controls">
            <motion.button
              className="speak-btn"
              onClick={speakWord}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              🔊 Hear Word
            </motion.button>
            <motion.button
              className="hint-btn"
              onClick={() => setShowHint(true)}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              disabled={showHint}
            >
              💡 Show Hint
            </motion.button>
          </div>

          {showHint && (
            <motion.div 
              className="hint"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              {currentWord.hint}
            </motion.div>
          )}

          <form onSubmit={handleSubmit} className="spelling-form">
            <input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="Type the word here..."
              autoFocus
            />
            <motion.button
              type="submit"
              className="submit-btn"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Check Spelling
            </motion.button>
          </form>

          {feedback && (
            <motion.div
              className={`feedback ${feedback.type}`}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
            >
              {feedback.message}
            </motion.div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default SpellingBee;