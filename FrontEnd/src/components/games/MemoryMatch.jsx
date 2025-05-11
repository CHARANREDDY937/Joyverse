import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './MemoryMatch.css';

const emotionBackgrounds = {
  Happiness: 'https://i.pinimg.com/736x/3c/c2/4c/3cc24c1323758ad3ac771422cca85b16.jpg',
  Sadness: 'https://i.pinimg.com/736x/af/a3/93/afa3935151761fafefe50b3b4cf4e22b.jpg',
  Anger: 'https://i.pinimg.com/736x/1b/c2/54/1bc254fc6ac4e9bc66c906b8e222c9e5.jpg',
  Surprise: 'https://i.pinimg.com/736x/b5/08/2c/b5082cfb446b91fde276b51692f61f8b.jpg',
  Disgust: 'https://i.pinimg.com/736x/e3/ed/87/e3ed8733e6a1ff0400821e2c829a11bd.jpg',
  Fear: 'https://i.pinimg.com/736x/86/b6/59/86b659584ccc8d660248fef17e6dad7b.jpg',
  Neutral: 'https://i.pinimg.com/736x/03/98/cb/0398cbb268528dbad35799ad602128be.jpg',
};

const instructionBackground = 'https://i.pinimg.com/736x/69/05/c3/6905c322607bf2b30486d2b19ba71de7.jpg';

const EMOJIS = ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨', '🐯', '🦁', '🐮', '🐷', '🐸', '🐵', '🐔', '🐧', '🐦', '🦄', '🐴'];

const generateCardLevel = (emojiSet, count) => {
  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    emoji: emojiSet[i % emojiSet.length],
    name: `animal-${i + 1}`,
  }));
};

const CARD_LEVELS = {
  easy: Array.from({ length: 20 }, (_, i) => generateCardLevel(EMOJIS, 2 + i)),
  medium: Array.from({ length: 20 }, (_, i) => generateCardLevel(EMOJIS, 10 + i)),
  hard: Array.from({ length: 20 }, (_, i) => generateCardLevel(EMOJIS, 15 + i)),
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
  }, [currentLevelIndex, currentSubLevel]);

  useEffect(() => {
    if (gameStarted) {
      initializeGame();
    }
  }, [gameStarted, initializeGame]);

  const handleCardClick = (clickedCard) => {
    if (flippedCards.length === 2) return;
    if (matchedPairs.includes(clickedCard.id) || 
        flippedCards.find((card) => card.uniqueId === clickedCard.uniqueId)) return;

    const newFlippedCards = [...flippedCards, clickedCard];
    setFlippedCards(newFlippedCards);

    if (newFlippedCards.length === 2) {
      setMoves((prev) => prev + 1);

      if (newFlippedCards[0].id === newFlippedCards[1].id) {
        setMatchedPairs((prev) => [...prev, clickedCard.id]);
        setScore((prev) => prev + 50);
        setFlippedCards([]);
      } else {
        setTimeout(() => setFlippedCards([]), 1000);
      }
    }
  };

  const handleNextLevel = () => {
    const level = levelOrder[currentLevelIndex];
    if (currentSubLevel + 1 < CARD_LEVELS[level].length) {
      setCurrentSubLevel(currentSubLevel + 1);
    } else if (currentLevelIndex + 1 < levelOrder.length) {
      setCurrentLevelIndex(currentLevelIndex + 1);
      setCurrentSubLevel(0);
    } else {
      setGameStarted(false);
      setCurrentLevelIndex(0);
      setCurrentSubLevel(0);
    }
  };

  const isCardFlipped = (card) => {
    return flippedCards.find((flipped) => flipped.uniqueId === card.uniqueId) ||
           matchedPairs.includes(card.id);
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
        onClick={() => navigate('/child/games')}
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
          <div className="step"><div className="step-number">5</div><p>Finish all levels to win!</p></div>
        </div>
        <motion.button
          className="start-game-btn"
          onClick={() => setGameStarted(true)}
          whileTap={{ scale: 0.95 }}
        >
          Let's Play
        </motion.button>
      </div>
    </div>
  );

  const GameScreen = () => {
    const levelName = levelOrder[currentLevelIndex];
    return (
      <div 
        className="memory-match"
        style={{
          background: `url(${emotionBackgrounds[emotion]})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
          minHeight: '100vh',
          width: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          padding: '2rem'
        }}
      >
        <div className="game-controls">
          <motion.button
            className="back-button"
            onClick={() => {
              setGameStarted(false);
              setCurrentLevelIndex(0);
              setCurrentSubLevel(0);
            }}
            whileTap={{ scale: 0.95 }}
          >
            ← Back to Instructions
          </motion.button>
        </div>

        <div className="game-info" style={{ background: 'rgba(255, 255, 255, 0.9)', padding: '1rem', borderRadius: '8px', marginBottom: '2rem' }}>
          <span className="moves">Moves: {moves}</span>
          <span className="score">Score: {score}</span>
          <motion.button 
            className="reset-button" 
            onClick={initializeGame} 
            whileTap={{ scale: 0.95 }}
          >
            Reset
          </motion.button>
        </div>

        <motion.div
          className="game-board"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
          style={{ 
            background: 'rgba(255, 255, 255, 0.8)',
            padding: '2rem',
            borderRadius: '12px',
            width: '100%',
            maxWidth: '900px'
          }}
        >
          {cards.map((card) => (
            <motion.div
              key={card.uniqueId}
              className={`card ${isCardFlipped(card) ? 'flipped' : ''}`}
              onClick={() => handleCardClick(card)}
              whileTap={{ scale: 0.95 }}
            >
              <div className="card-inner">
                <div className="card-front"><span>?</span></div>
                <div className="card-back"><span>{card.emoji}</span></div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {matchedPairs.length === CARD_LEVELS[levelName][currentSubLevel].length && (
          <motion.div
            className="game-complete"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
          >
            <h2>Level {currentSubLevel + 1} Complete! 🎉</h2>
            <p>{levelName.toUpperCase()} Level in {moves} moves!</p>
            <motion.button
              className="play-again-btn"
              onClick={handleNextLevel}
              whileTap={{ scale: 0.95 }}
            >
              {currentLevelIndex === 2 && currentSubLevel === 19
                ? 'Finish Game'
                : 'Next Level'}
            </motion.button>
          </motion.div>
        )}
      </div>
    );
  };

  return (
    <AnimatePresence mode="wait">
      {gameStarted ? <GameScreen /> : <InstructionsScreen />}
    </AnimatePresence>
  );
};

export default MemoryMatch;