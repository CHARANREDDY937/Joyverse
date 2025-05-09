import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './PuzzleWorld.css';

const PUZZLES = [
  {
    id: 1,
    name: 'Forest Friends',
    image: 'https://i.pinimg.com/736x/88/95/7b/88957bd3d472ea7e417f92a9b24f0cbb.jpg',
    size: 3 // 3x3 grid
  },
  {
    id: 2,
    name: 'Ocean Adventure',
    image: 'https://i.pinimg.com/736x/d8/bb/c8/d8bbc8635fbdaafdce23e3a3306a3f84.jpg',
    size: 3
  },
  {
    id: 3,
    name: 'Space Journey',
    image: 'https://i.pinimg.com/736x/a3/c6/e0/a3c6e0e88875cadaea284c8524f8d989.jpg',
    size: 3
  }
];

// Emotion-to-background mapping (same as WordWizard, MathSafari, and MemoryMatch)
const emotionBackgrounds = {
  Happiness: 'https://i.pinimg.com/736x/3c/c2/4c/3cc24c1323758ad3ac771422cca85b16.jpg',
  Sadness: 'https://i.pinimg.com/736x/af/a3/93/afa3935151761fafefe50b3b4cf4e22b.jpg',
  Anger: 'https://i.pinimg.com/736x/1b/c2/54/1bc254fc6ac4e9bc66c906b8e222c9e5.jpg',
  Surprise: 'https://i.pinimg.com/736x/b5/08/2c/b5082cfb446b91fde276b51692f61f8b.jpg',
  Disgust: 'https://i.pinimg.com/736x/e3/ed/87/e3ed8733e6a1ff0400821e2c829a11bd.jpg',
  Fear: 'https://i.pinimg.com/736x/86/b6/59/86b659584ccc8d660248fef17e6dad7b.jpg',
  Neutral: 'https://i.pinimg.com/736x/03/98/cb/0398cbb268528dbad35799ad602128be.jpg',
};

const PuzzleWorld = ({ emotion = 'Neutral' }) => {
  const navigate = useNavigate();
  const [currentPuzzle, setCurrentPuzzle] = useState(PUZZLES[0]);
  const [tiles, setTiles] = useState([]);
  const [moves, setMoves] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState(null);

  const isSolvable = useCallback((tiles) => {
    let inversions = 0;
    const size = currentPuzzle.size;

    for (let i = 0; i < tiles.length - 1; i++) {
      if (!tiles[i]) continue;
      for (let j = i + 1; j < tiles.length; j++) {
        if (!tiles[j]) continue;
        if (tiles[i] > tiles[j]) inversions++;
      }
    }

    const emptyRowFromBottom = Math.floor(tiles.indexOf(null) / size);
    return (size % 2 === 1) ? 
      (inversions % 2 === 0) : 
      ((inversions + emptyRowFromBottom) % 2 === 0);
  }, [currentPuzzle.size]);

  const initializePuzzle = useCallback(() => {
    // Clear emotions logs on backend
    fetch('http://localhost:8000/clear_emotions_log', {
      method: 'POST',
    })
      .then(response => response.json())
      .then(data => {
        if (data.status !== 'success') {
          setError('Failed to clear emotions and percentages logs: ' + data.message);
        } else {
          setError(null);
        }
      })
      .catch(error => {
        setError('Error clearing emotions and percentages logs: ' + error.message);
      });

    const shuffleTiles = (tiles) => {
      const shuffled = [...tiles];
      let currentIndex = shuffled.length;

      while (currentIndex !== 0) {
        const randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        [shuffled[currentIndex], shuffled[randomIndex]] = 
          [shuffled[randomIndex], shuffled[currentIndex]];
      }

      if (!isSolvable(shuffled)) {
        const lastIndex = shuffled.length - 1;
        if (lastIndex >= 2) {
          [shuffled[lastIndex - 1], shuffled[lastIndex - 2]] = 
            [shuffled[lastIndex - 2], shuffled[lastIndex - 1]];
        }
      }

      return shuffled;
    };

    const size = currentPuzzle.size * currentPuzzle.size;
    const newTiles = Array.from({ length: size - 1 }, (_, i) => i + 1);
    newTiles.push(null); // Empty tile
    
    const shuffledTiles = shuffleTiles(newTiles);
    setTiles(shuffledTiles);
    setMoves(0);
    setIsComplete(false);
  }, [currentPuzzle, isSolvable]);

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

  useEffect(() => {
    initializePuzzle();

    // Cleanup: Append percentages when component unmounts
    return () => {
      appendEmotionPercentages();
      console.log('PuzzleWorld unmounted: Emotion percentages appended.');
    };
  }, [currentPuzzle, initializePuzzle]);

  const handleTileClick = (index) => {
    if (isComplete) return;

    const size = currentPuzzle.size;
    const emptyIndex = tiles.indexOf(null);
    
    const isAdjacent = (
      (Math.abs(index - emptyIndex) === 1 && Math.floor(index / size) === Math.floor(emptyIndex / size)) ||
      Math.abs(index - emptyIndex) === size
    );

    if (isAdjacent) {
      const newTiles = [...tiles];
      [newTiles[index], newTiles[emptyIndex]] = [newTiles[emptyIndex], newTiles[index]];
      setTiles(newTiles);
      setMoves(moves + 1);

      const isComplete = newTiles.every((tile, index) => 
        tile === null ? index === newTiles.length - 1 : tile === index + 1
      );
      setIsComplete(isComplete);
    }
  };

  const handleBack = async () => {
    // Append percentages before navigating away
    await appendEmotionPercentages();
    navigate('/child/games');
  };

  const backgroundImage = emotionBackgrounds[emotion] || emotionBackgrounds.Neutral;

  return (
    <div 
      className="puzzle-world" 
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
        <h1>Puzzle World</h1>
        <div className="game-info">
          <span className="moves">Moves: {moves}</span>
          <motion.button
            className="reset-btn"
            onClick={initializePuzzle}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Reset Puzzle
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

      <div className="puzzle-selector">
        {PUZZLES.map(puzzle => (
          <motion.button
            key={puzzle.id}
            className={`puzzle-btn ${currentPuzzle.id === puzzle.id ? 'active' : ''}`}
            onClick={() => setCurrentPuzzle(puzzle)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {puzzle.name}
          </motion.button>
        ))}
      </div>

      <motion.div 
        className="puzzle-container"
        style={{ '--grid-size': currentPuzzle.size }}
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
      >
        {tiles.map((tile, index) => (
          <motion.div
            key={index}
            className={`puzzle-tile ${!tile ? 'empty' : ''}`}
            onClick={() => handleTileClick(index)}
            whileHover={{ scale: tile ? 1.05 : 1 }}
            whileTap={{ scale: tile ? 0.95 : 1 }}
            style={{
              backgroundImage: tile ? `url(${currentPuzzle.image})` : 'none',
              backgroundSize: `${currentPuzzle.size * 100}%`,
              backgroundPosition: tile ? 
                `${((tile - 1) % currentPuzzle.size) * -100}% ${Math.floor((tile - 1) / currentPuzzle.size) * -100}%` 
                : 'center'
            }}
          >
            {tile}
          </motion.div>
        ))}
      </motion.div>

      {isComplete && (
        <motion.div 
          className="victory-screen"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
        >
          <h2>Congratulations! 🎉</h2>
          <p>You solved the puzzle in {moves} moves!</p>
          <motion.button
            className="next-puzzle-btn"
            onClick={() => {
              const nextIndex = (PUZZLES.indexOf(currentPuzzle) + 1) % PUZZLES.length;
              setCurrentPuzzle(PUZZLES[nextIndex]);
            }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Next Puzzle
          </motion.button>
        </motion.div>
      )}
    </div>
  );
};

export default PuzzleWorld;