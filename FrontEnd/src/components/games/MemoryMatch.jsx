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

// Emotion-to-background mapping (same as WordWizard and MathSafari)
const emotionBackgrounds = {
  Happiness: 'https://i.pinimg.com/736x/3c/c2/4c/3cc24c1323758ad3ac771422cca85b16.jpg',
  Sadness: 'https://i.pinimg.com/736x/af/a3/93/afa3935151761fafefe50b3b4cf4e22b.jpg',
  Anger: 'https://i.pinimg.com/736x/1b/c2/54/1bc254fc6ac4e9bc66c906b8e222c9e5.jpg',
  Surprise: 'https://i.pinimg.com/736x/b5/08/2c/b5082cfb446b91fde276b51692f61f8b.jpg',
  Disgust: 'https://i.pinimg.com/736x/e3/ed/87/e3ed8733e6a1ff0400821e2c829a11bd.jpg',
  Fear: 'https://i.pinimg.com/736x/86/b6/59/86b659584ccc8d660248fef17e6dad7b.jpg',
  Neutral: 'https://i.pinimg.com/736x/03/98/cb/0398cbb268528dbad35799ad602128be.jpg',
};

const MemoryMatch = ({ emotion = 'Neutral' }) => {
  const navigate = useNavigate();
  const [cards, setCards] = useState([]);
  const [flippedCards, setFlippedCards] = useState([]);
  const [matchedPairs, setMatchedPairs] = useState([]);
  const [moves, setMoves] = useState(0);
  const [score, setScore] = useState(0);
  const [error, setError] = useState(null);

  // Initialize game and clear emotions logs
  const initializeGame = async () => {
    try {
      const response = await fetch('http://localhost:8000/clear_emotions_log', {
        method: 'POST',
      });
      const data = await response.json();
      if (data.status !== 'success') {
        setError('Failed to clear emotions and percentages logs: ' + data.message);
      } else {
        setError(null);
      }
    } catch (error) {
      setError('Error clearing emotions and percentages logs: ' + error.message);
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

  // Append emotion percentages
  const appendEmotionPercentages = async () => {
    try {
      const response = await fetch('http://localhost:8000/append_emotion_percentages', {
        method: 'POST',
      });
      const data = await response.json();
      if (data.status !== 'success') {
        setError('Failed to append emotion percentages: ' + data.message);
      } else {
        setError(null);
      }
    } catch (error) {
      setError('Error appending emotion percentages: ' + error.message);
    }
  };

  // UseEffect to initialize game
  useEffect(() => {
    initializeGame();

    // Cleanup: Append percentages when component unmounts
    return () => {
      appendEmotionPercentages();
      console.log('MemoryMatch unmounted: Emotion percentages appended.');
    };
  }, []);

  const handleCardClick = (clickedCard) => {
    if (flippedCards.length === 2) return;
    
    if (matchedPairs.includes(clickedCard.id) || 
        flippedCards.find(card => card.uniqueId === clickedCard.uniqueId)) return;

    const newFlippedCards = [...flippedCards, clickedCard];
    setFlippedCards(newFlippedCards);

    if (newFlippedCards.length === 2) {
      setMoves(moves => moves + 1);

      if (newFlippedCards[0].id === newFlippedCards[1].id) {
        setMatchedPairs([...matchedPairs, clickedCard.id]);
        setScore(score => score + 50);
        setFlippedCards([]);
      } else {
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

  const handleBack = async () => {
    // Append percentages before navigating away
    await appendEmotionPercentages();
    navigate('/child/games');
  };

  const backgroundImage = emotionBackgrounds[emotion] || emotionBackgrounds.Neutral;

  return (
    <div 
      className="memory-match" 
      style={{ 
        backgroundImage: `url(${backgroundImage})`,
      }}
    >
      <motion.button
        className="back-button"
        onClick={handleBack}
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
          <motion.button
            className="reset-btn"
            onClick={initializeGame}
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