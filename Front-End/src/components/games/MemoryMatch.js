import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import './MemoryMatch.css';

const emojis = ['🐶', '🐱', '🐰', '🐼', '🦊', '🦁', '🐯', '🐮'];
const cardVariants = {
  hidden: { rotateY: 180 },
  visible: { rotateY: 0 },
};

const MemoryMatch = () => {
  const [cards, setCards] = useState([]);
  const [flippedIndices, setFlippedIndices] = useState([]);
  const [matchedPairs, setMatchedPairs] = useState([]);
  const [moves, setMoves] = useState(0);
  const [gameComplete, setGameComplete] = useState(false);

  useEffect(() => {
    initializeGame();
  }, []);

  const initializeGame = () => {
    // Create pairs of emojis and shuffle them
    const emojiPairs = [...emojis, ...emojis];
    const shuffledEmojis = emojiPairs.sort(() => Math.random() - 0.5);
    setCards(shuffledEmojis);
    setFlippedIndices([]);
    setMatchedPairs([]);
    setMoves(0);
    setGameComplete(false);
  };

  const handleCardClick = (index) => {
    // Prevent clicking if two cards are already flipped or same card is clicked
    if (flippedIndices.length === 2 || flippedIndices.includes(index) || matchedPairs.includes(index)) {
      return;
    }

    const newFlippedIndices = [...flippedIndices, index];
    setFlippedIndices(newFlippedIndices);

    // Check for match when two cards are flipped
    if (newFlippedIndices.length === 2) {
      setMoves(moves + 1);
      
      if (cards[newFlippedIndices[0]] === cards[newFlippedIndices[1]]) {
        // Match found
        setMatchedPairs([...matchedPairs, ...newFlippedIndices]);
        setFlippedIndices([]);

        // Check if game is complete
        if (matchedPairs.length + 2 === cards.length) {
          setGameComplete(true);
        }
      } else {
        // No match - flip cards back after delay
        setTimeout(() => {
          setFlippedIndices([]);
        }, 1000);
      }
    }
  };

  return (
    <div className="memory-game">
      <div className="game-header">
        <h1>Memory Match</h1>
        <div className="game-info">
          <span>Moves: {moves}</span>
          <button onClick={initializeGame} className="reset-button">
            New Game
          </button>
        </div>
      </div>

      <div className="cards-grid">
        {cards.map((emoji, index) => (
          <motion.div
            key={index}
            className={`card ${
              flippedIndices.includes(index) || matchedPairs.includes(index) ? 'flipped' : ''
            }`}
            onClick={() => handleCardClick(index)}
            variants={cardVariants}
            initial="hidden"
            animate={flippedIndices.includes(index) || matchedPairs.includes(index) ? 'visible' : 'hidden'}
            transition={{ duration: 0.6 }}
          >
            <div className="card-inner">
              <div className="card-front">❓</div>
              <div className="card-back">{emoji}</div>
            </div>
          </motion.div>
        ))}
      </div>

      {gameComplete && (
        <motion.div
          className="game-complete"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ duration: 0.5 }}
        >
          <h2>🎉 Congratulations! 🎉</h2>
          <p>You completed the game in {moves} moves!</p>
          <button onClick={initializeGame}>Play Again</button>
        </motion.div>
      )}
    </div>
  );
};

export default MemoryMatch; 