import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import './WordWizard.css';
import { useNavigate } from 'react-router-dom';

// Move categories outside component to prevent recreation on each render
const WORD_CATEGORIES = {
  animals: ['ELEPHANT', 'GIRAFFE', 'PENGUIN', 'DOLPHIN', 'KANGAROO'],
  fruits: ['APPLE', 'BANANA', 'ORANGE', 'MANGO', 'STRAWBERRY'],
  countries: ['FRANCE', 'JAPAN', 'BRAZIL', 'INDIA', 'CANADA']
};

const WordWizard = () => {
  const navigate = useNavigate();
  const [word, setWord] = useState('');
  const [guessedLetters, setGuessedLetters] = useState(new Set());
  const [remainingLives, setRemainingLives] = useState(6);
  const [gameStatus, setGameStatus] = useState('playing'); // 'playing', 'won', 'lost'
  const [score, setScore] = useState(0);
  const [category, setCategory] = useState('animals');

  // Memoize the categories if they need to be transformed or filtered
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

    // Check win/lose conditions
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

  const handleBack = () => {
    navigate('/child/games');
  };

  return (
    <div className="word-wizard">
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
        </div>
      </div>

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
  );
};

export default WordWizard; 