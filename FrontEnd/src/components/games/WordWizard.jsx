import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import './WordWizard.css';
import { useNavigate } from 'react-router-dom';

// Emotion-to-background mapping
const emotionBackgrounds = {
  Happiness: 'https://i.pinimg.com/736x/3c/c2/4c/3cc24c1323758ad3ac771422cca85b16.jpg', // Sunny flowers
  Sadness: 'https://i.pinimg.com/736x/af/a3/93/afa3935151761fafefe50b3b4cf4e22b.jpg', // Rainy window
  Anger: 'https://i.pinimg.com/736x/1b/c2/54/1bc254fc6ac4e9bc66c906b8e222c9e5.jpg', // Stormy clouds
  Surprise: 'https://i.pinimg.com/736x/b5/08/2c/b5082cfb446b91fde276b51692f61f8b.jpg', // Colorful balloons
  Disgust: 'https://i.pinimg.com/736x/e3/ed/87/e3ed8733e6a1ff0400821e2c829a11bd.jpg', // Dark forest
  Fear: 'https://i.pinimg.com/736x/86/b6/59/86b659584ccc8d660248fef17e6dad7b.jpg', // Misty forest
  Neutral: 'https://i.pinimg.com/736x/03/98/cb/0398cbb268528dbad35799ad602128be.jpg', // Calm lake
};

const WORD_CATEGORIES = {
  animals: ['ELEPHANT', 'GIRAFFE', 'PENGUIN', 'DOLPHIN', 'KANGAROO'],
  fruits: ['APPLE', 'BANANA', 'ORANGE', 'MANGO', 'STRAWBERRY'],
  countries: ['FRANCE', 'JAPAN', 'BRAZIL', 'INDIA', 'CANADA']
};

const WordWizard = ({ emotion = 'Neutral' }) => {
  const navigate = useNavigate();
  const [word, setWord] = useState('');
  const [guessedLetters, setGuessedLetters] = useState(new Set());
  const [remainingLives, setRemainingLives] = useState(6);
  const [gameStatus, setGameStatus] = useState('playing');
  const [score, setScore] = useState(0);
  const [category, setCategory] = useState('animals');
  const [error, setError] = useState(null);

  const categories = useMemo(() => WORD_CATEGORIES, []);

  const selectNewWord = useCallback(() => {
    const words = categories[category];
    const randomWord = words[Math.floor(Math.random() * words.length)];
    setWord(randomWord);
    setGuessedLetters(new Set());
    setRemainingLives(6);
    setGameStatus('playing');
  }, [category, categories]);

  useEffect(() => {
    selectNewWord();
  }, [selectNewWord]);

  const handleLetterGuess = (letter) => {
    if (gameStatus !== 'playing') return;

    const newGuessedLetters = new Set(guessedLetters);
    newGuessedLetters.add(letter);
    setGuessedLetters(newGuessedLetters);

    if (!word.includes(letter)) {
      setRemainingLives(lives => lives - 1);
    }

    checkGameStatus(newGuessedLetters);
  };

  const checkGameStatus = (newGuessedLetters) => {
    if (word.split('').every(letter => newGuessedLetters.has(letter))) {
      setGameStatus('won');
      setScore(score => score + 100);
    } else if (remainingLives <= 1) {
      setGameStatus('lost');
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
    setCategory('animals');
    setGuessedLetters(new Set());
    setRemainingLives(6);
    setGameStatus('playing');
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
      console.log('WordWizard unmounted: Emotion logging should stop.');
    };
  }, []);

  const handleBack = () => {
    navigate('/child/games');
  };

  const backgroundImage = emotionBackgrounds[emotion] || emotionBackgrounds.Neutral;

  return (
    <div 
      className="word-wizard" 
      style={{ 
        backgroundImage: `url(${backgroundImage})`,
      }}
    >
      <div className="content-wrapper">
        <motion.button
          className="back-button"
          onClick={handleBack}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          ← Back to Games
        </motion.button>

        <div className="game-header">
          <h1>Word Wizard</h1>
          <div className="game-info">
            <span className="score">Score: {score}</span>
            <span className="lives">Lives: {'❤️'.repeat(remainingLives)}</span>
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

        <div className="category-selector">
          {Object.keys(categories).map(cat => (
            <motion.button
              key={cat}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className={`category-btn ${category === cat ? 'active' : ''}`}
              onClick={() => setCategory(cat)}
            >
              {cat.charAt(0).toUpperCase() + cat.slice(1)}
            </motion.button>
          ))}
        </div>

        <div className="word-display">
          {word.split('').map((letter, index) => (
            <motion.div
              key={index}
              className="letter-box"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: index * 0.1 }}
            >
              {guessedLetters.has(letter) ? letter : '_'}
            </motion.div>
          ))}
        </div>

        <div className="keyboard">
          {Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ').map((letter) => (
            <motion.button
              key={letter}
              className={`letter-btn ${guessedLetters.has(letter) ? 'used' : ''}`}
              onClick={() => handleLetterGuess(letter)}
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
              disabled={guessedLetters.has(letter) || gameStatus !== 'playing'}
            >
              {letter}
            </motion.button>
          ))}
        </div>

        {gameStatus !== 'playing' && (
          <motion.div
            className="game-over"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
          >
            <h2>{gameStatus === 'won' ? '🎉 Congratulations!' : '😢 Game Over'}</h2>
            <p>The word was: {word}</p>
            <motion.button
              className="play-again-btn"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={selectNewWord}
            >
              Play Again
            </motion.button>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default WordWizard;