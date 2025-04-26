import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import './Hangman.css';

const words = {
  animals: [
    { word: 'ELEPHANT', hint: 'A large mammal with a trunk' },
    { word: 'GIRAFFE', hint: 'Tallest animal with a long neck' },
    { word: 'PENGUIN', hint: 'A bird that cannot fly but swims well' },
    { word: 'DOLPHIN', hint: 'An intelligent marine mammal' },
    { word: 'BUTTERFLY', hint: 'A colorful insect with wings' },
    { word: 'KANGAROO', hint: 'An Australian animal that hops' },
  ],
  fruits: [
    { word: 'APPLE', hint: 'A round fruit that keeps the doctor away' },
    { word: 'BANANA', hint: 'A long, yellow fruit rich in potassium' },
    { word: 'ORANGE', hint: 'A citrus fruit with vitamin C' },
    { word: 'STRAWBERRY', hint: 'A red fruit with tiny seeds on the outside' },
    { word: 'PINEAPPLE', hint: 'A tropical fruit with spiky skin' },
    { word: 'MANGO', hint: 'A sweet and juicy tropical fruit' },
  ],
  colors: [
    { word: 'PURPLE', hint: 'A mix of red and blue' },
    { word: 'YELLOW', hint: 'The color of the sun' },
    { word: 'ORANGE', hint: 'A color and a fruit' },
    { word: 'GREEN', hint: 'The color of grass' },
    { word: 'BLUE', hint: 'The color of the sky' },
    { word: 'PINK', hint: 'A soft and romantic color' },
  ],
};

const Hangman = () => {
  const [wordIndex, setWordIndex] = useState(0);
  const [category, setCategory] = useState('');
  const [guessedLetters, setGuessedLetters] = useState(new Set());
  const [wrongGuesses, setWrongGuesses] = useState(0);
  const [gameStatus, setGameStatus] = useState('category'); // category, playing, won, lost
  const maxWrongGuesses = 6;

  const selectCategory = (selectedCategory) => {
    setCategory(selectedCategory);
    setWordIndex(0);
    setGuessedLetters(new Set());
    setWrongGuesses(0);
    setGameStatus('playing');
  };

  const guessLetter = (letter) => {
    if (gameStatus !== 'playing') return;

    const newGuessedLetters = new Set(guessedLetters);
    newGuessedLetters.add(letter);
    setGuessedLetters(newGuessedLetters);

    if (!words[category][wordIndex].word.includes(letter)) {
      const newWrongGuesses = wrongGuesses + 1;
      setWrongGuesses(newWrongGuesses);
      if (newWrongGuesses >= maxWrongGuesses) {
        setGameStatus('lost');
      }
    }
  };

  useEffect(() => {
    if (gameStatus === 'playing' && words[category]) {
      const isWon = [...words[category][wordIndex].word].every(letter => guessedLetters.has(letter));
      if (isWon) {
        if (wordIndex < words[category].length - 1) {
          setWordIndex(wordIndex + 1);
          setGuessedLetters(new Set());
          setWrongGuesses(0);
        } else {
          setGameStatus('won');
        }
      }
    }
  }, [guessedLetters, category, wordIndex, gameStatus]);

  const renderWord = () => {
    return words[category][wordIndex].word.split('').map((letter, index) => (
      <motion.span
        key={index}
        className="letter-box"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ delay: index * 0.1 }}
      >
        {guessedLetters.has(letter) ? letter : '_'}
      </motion.span>
    ));
  };

  const renderKeyboard = () => {
    const keyboard = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    return keyboard.map((letter) => (
      <motion.button
        key={letter}
        className={`keyboard-letter ${guessedLetters.has(letter) ? 'used' : ''}`}
        onClick={() => guessLetter(letter)}
        disabled={guessedLetters.has(letter) || gameStatus !== 'playing'}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
      >
        {letter}
      </motion.button>
    ));
  };

  if (gameStatus === 'category') {
    return (
      <div className="hangman-game">
        <h1>Choose a Category</h1>
        <div className="category-selection">
          {Object.keys(words).map((cat) => (
            <motion.button
              key={cat}
              className="category-button"
              onClick={() => selectCategory(cat)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {cat.charAt(0).toUpperCase() + cat.slice(1)}
            </motion.button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="hangman-game">
      <div className="game-header">
        <h1>Word Adventure</h1>
        <div className="game-info">
          <span>Category: {category}</span>
          <span>Wrong Guesses: {wrongGuesses}/{maxWrongGuesses}</span>
        </div>
      </div>

      <div className="word-container">
        {renderWord()}
      </div>

      <p className="hint">Hint: {words[category][wordIndex].hint}</p>

      <div className="keyboard-container">
        {renderKeyboard()}
      </div>

      {gameStatus === 'won' && (
        <motion.div
          className="game-result"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <h2>🎉 You Completed All Words! 🎉</h2>
        </motion.div>
      )}
    </div>
  );
};

export default Hangman;