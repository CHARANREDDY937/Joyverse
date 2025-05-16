import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './MemoryMatch.css';

const EMOJIS = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨', '🐯', '🦁', '🐮', '🐷', '🐸', '🐵', '🐔', '🐧', '🐦', '🦄', '🐴'];

const emotionBackgrounds = {
  Happiness: 'https://i.pinimg.com/736x/24/27/1a/24271ab068c02d02efc87e27888405ca.jpg',
  Sadness: 'https://i.pinimg.com/736x/af/a3/93/afa3935151761fafefe50b3b4cf4e22b.jpg',
  Anger: 'https://i.pinimg.com/736x/1b/c2/54/1bc254fc6ac4e9bc66c906b8e222c9e5.jpg',
  Surprise: 'https://i.pinimg.com/736x/b5/08/2c/b5082cfb446b91fde276b51692f61f8b.jpg',
  Disgust: 'https://i.pinimg.com/736x/e3/ed/87/e3ed8733e6a1ff0400821e2c829a11bd.jpg',
  Fear: 'https://i.pinimg.com/736x/86/b6/59/86b659584ccc8d660248fef17e6dad7b.jpg',
  Neutral: 'https://i.pinimg.com/736x/24/27/1a/24271ab068c02d02efc87e27888405ca.jpg',
};

const instructionBackground = 'https://i.pinimg.com/736x/69/05/c3/6905c322607bf2b30486d2b19ba71de7.jpg';

const generateCardLevel = (emojiSet, count) => {
  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    emoji: emojiSet[i % emojiSet.length],
    name: `animal-${i + 1}`,
  }));
};

const CARD_LEVELS = {
  easy: Array.from({ length: 3 }, (_, i) => generateCardLevel(EMOJIS, 4 + i)),
  medium: Array.from({ length: 3 }, (_, i) => generateCardLevel(EMOJIS, 6 + i)),
  hard: Array.from({ length: 3 }, (_, i) => generateCardLevel(EMOJIS, 8 + i)),
};

const levelOrder = ['easy', 'medium', 'hard'];

const MemoryMatch = ({ emotion = 'Neutral' }) => {
  const navigate = useNavigate();
  const [gameStarted, setGameStarted] = useState(false);
  const [cards, setCards] = useState([]);
  const [flippedCards, setFlippedCards] = useState([]);
  const [matchedPairs, setMatchedPairs] = useState([]);
  const [moves, setMoves] = useState(0);
  const [score, setScore] = useState(0);
  const [currentLevelIndex, setCurrentLevelIndex] = useState(0);
  const [currentSubLevel, setCurrentSubLevel] = useState(0);
  const [error, setError] = useState(null);

  const initializeGame = useCallback(() => {
    const level = levelOrder[currentLevelIndex];
    const selectedCards = CARD_LEVELS[level][currentSubLevel];
    const shuffledCards = [...selectedCards, ...selectedCards]
      .sort(() => Math.random() - 0.5)
      .map((card, index) => ({
        ...card,
        uniqueId: `${card.id}-${index}`,
      }));

    setCards(shuffledCards);
    setFlippedCards([]);
    setMatchedPairs([]);
    setMoves(0);
    setScore(0);

    fetch('http://localhost:8000/clear_emotions_log', {
      method: 'POST',
    }).catch(error => {
      console.log('Non-critical error clearing emotions log:', error);
    });
  }, [currentLevelIndex, currentSubLevel]);

  const appendEmotionPercentages = async () => {
    try {
      await fetch('http://localhost:8000/append_emotion_percentages', {
        method: 'POST',
      });
    } catch (error) {
      console.log('Non-critical error appending emotions:', error);
    }
  };

  useEffect(() => {
    if (gameStarted) {
      initializeGame();
    }
  }, [gameStarted, initializeGame]);

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

        // Check if level is complete
        if (matchedPairs.length + 1 === cards.length / 2) {
          setTimeout(() => {
            handleNextLevel();
          }, 1000);
        }
      } else {
        setTimeout(() => {
          setFlippedCards([]);
        }, 1000);
      }
    }
  };

  const handleNextLevel = () => {
    const level = levelOrder[currentLevelIndex];
    if (currentSubLevel + 1 < CARD_LEVELS[level].length) {
      setCurrentSubLevel(prev => prev + 1);
    } else if (currentLevelIndex + 1 < levelOrder.length) {
      setCurrentLevelIndex(prev => prev + 1);
      setCurrentSubLevel(0);
    } else {
      setGameStarted(false);
      setCurrentLevelIndex(0);
      setCurrentSubLevel(0);
    }
  };

  const isCardFlipped = (card) => {
    return flippedCards.find(flipped => flipped.uniqueId === card.uniqueId) ||
           matchedPairs.includes(card.id);
  };

  const handleBack = async () => {
    await appendEmotionPercentages();
    navigate('/child/games');
  };

  const InstructionsScreen = () => (
    <div 
      className="instructions-screen"
      style={{
        backgroundImage: `url(${instructionBackground})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
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
      <div className="instructions-content">
        <h1>Memory Match</h1>
        <h2>How to Play</h2>
        <div className="instruction-steps">
          <div className="step"><div className="step-number">1</div><p>Match pairs of animal cards</p></div>
          <div className="step"><div className="step-number">2</div><p>Click any two cards to flip them</p></div>
          <div className="step"><div className="step-number">3</div><p>If they match, they stay face up</p></div>
          <div className="step"><div className="step-number">4</div><p>If not, they flip back</p></div>
          <div className="step"><div className="step-number">5</div><p>Complete all levels to win!</p></div>
        </div>
        <motion.button
          className="start-game-btn"
          onClick={() => setGameStarted(true)}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Let's Play!
        </motion.button>
      </div>
    </div>
  );

  const backgroundImage = emotionBackgrounds[emotion] || emotionBackgrounds.Neutral;

  return !gameStarted ? (
    <InstructionsScreen />
  ) : (
    <div 
      className="memory-match" 
      style={{ 
        backgroundImage: `url(${backgroundImage})`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat',
        minHeight: '100vh'
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
        <div className="level-info">
          <h2>Level: {levelOrder[currentLevelIndex].toUpperCase()} - {currentSubLevel + 1}</h2>
        </div>
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

      <div className="game-board">
        {cards.map((card) => (
          <motion.div
            key={card.uniqueId}
            className={`card ${isCardFlipped(card) ? 'flipped' : ''}`}
            onClick={() => handleCardClick(card)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <div className="card-inner">
              <div className="card-front">❓</div>
              <div className="card-back">{card.emoji}</div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default MemoryMatch;