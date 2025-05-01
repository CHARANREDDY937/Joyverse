import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './MemoryMatch.css';

const CARD_PAIRS = [
  { id: 1, emoji: '🐘', name: 'elephant' },
  { id: 2, emoji: '🦒', name: 'giraffe' },
  { id: 3, emoji: '🦁', name: 'lion' },
  { id: 4, emoji: '🐯', name: 'tiger' },
  { id: 5, emoji: '🐼', name: 'panda' },
  { id: 6, emoji: '🦊', name: 'fox' },
  { id: 7, emoji: '🐨', name: 'koala' },
  { id: 8, emoji: '🦘', name: 'kangaroo' }
];

const MemoryMatch = () => {
  const navigate = useNavigate();
  const [cards, setCards] = useState([]);
  const [flippedCards, setFlippedCards] = useState([]);
  const [matchedPairs, setMatchedPairs] = useState([]);
  const [moves, setMoves] = useState(0);
  const [score, setScore] = useState(0);
  const [error, setError] = useState(null);

  // Initialize game
  const initializeGame = async () => {
    // Clear previous emotions log on backend
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

    // Create pairs of cards and shuffle them
    const shuffledCards = [...CARD_PAIRS, ...CARD_PAIRS]
      .sort(() => Math.random() - 0.5)
      .map((card, index) => ({
        ...card,
        uniqueId: `${card.id}-${index}`
      }));
    
    setCards(shuffledCards);
    setFlippedCards([]);
    setMatchedPairs([]);
    setMoves(0);
    setScore(0);
  };

  // UseEffect to initialize game
  useEffect(() => {
    initializeGame();
  }, []);

  const handleCardClick = (clickedCard) => {
    // Prevent clicking if two cards are already flipped
    if (flippedCards.length === 2) return;
    
    // Prevent clicking on already matched or flipped cards
    if (matchedPairs.includes(clickedCard.id) || 
        flippedCards.find(card => card.uniqueId === clickedCard.uniqueId)) return;

    const newFlippedCards = [...flippedCards, clickedCard];
    setFlippedCards(newFlippedCards);

    // If two cards are flipped, check for match
    if (newFlippedCards.length === 2) {
      setMoves(moves => moves + 1);

      if (newFlippedCards[0].id === newFlippedCards[1].id) {
        // Match found
        setMatchedPairs([...matchedPairs, clickedCard.id]);
        setScore(score => score + 50);
        setFlippedCards([]);
      } else {
        // No match, flip cards back
        setTimeout(() => {
          setFlippedCards([]);
        }, 1000);
      }
    }
  };

  const isCardFlipped = (card) => {
    return flippedCards.find(flipped => flipped.uniqueId === card.uniqueId) ||
           matchedPairs.includes(card.id);
  };

  return (
    <div className="memory-match">
      <motion.button
        className="back-button"
        onClick={() => navigate('/child/games')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Games
      </motion.button>

      <div className="game-header">
        <h1>Memory Match</h1>
        <div className="game-info">
          <span className="moves">Moves: {moves}</span>
          <span className="score">Score: {score}</span>
          <button className="reset-btn" onClick={initializeGame}>
            Reset Game
          </button>
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

      <motion.div 
        className="game-board"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        {cards.map((card) => (
          <motion.div
            key={card.uniqueId}
            className={`card ${isCardFlipped(card) ? 'flipped' : ''}`}
            onClick={() => handleCardClick(card)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <div className="card-inner">
              <div className="card-front">
                <span>?</span>
              </div>
              <div className="card-back">
                <span>{card.emoji}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </motion.div>

      {matchedPairs.length === CARD_PAIRS.length && (
        <motion.div 
          className="game-complete"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
        >
          <h2>Congratulations! 🎉</h2>
          <p>You completed the game in {moves} moves!</p>
          <motion.button
            className="play-again-btn"
            onClick={initializeGame}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Play Again
          </motion.button>
        </motion.div>
      )}
    </div>
  );
};

export default MemoryMatch;