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

const PuzzleWorld = () => {
  const navigate = useNavigate();
  const [currentPuzzle, setCurrentPuzzle] = useState(PUZZLES[0]);
  const [tiles, setTiles] = useState([]);
  const [moves, setMoves] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

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
    // Clear previous emotions log on backend
    fetch('http://localhost:8000/clear_emotions_log', {
      method: 'POST',
    })
      .then(response => response.json())
      .then(data => {
        if (data.status !== 'success') {
          console.error('Failed to clear emotions log:', data.message);
        }
      })
      .catch(error => {
        console.error('Error clearing emotions log:', error);
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

      // Ensure puzzle is solvable
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

  useEffect(() => {
    initializePuzzle();
  }, [currentPuzzle, initializePuzzle]); // Added currentPuzzle to dependencies

  const handleTileClick = (index) => {
    if (isComplete) return;

    const size = currentPuzzle.size;
    const emptyIndex = tiles.indexOf(null);
    
    // Check if clicked tile is adjacent to empty space
    const isAdjacent = (
      (Math.abs(index - emptyIndex) === 1 && Math.floor(index / size) === Math.floor(emptyIndex / size)) ||
      Math.abs(index - emptyIndex) === size
    );

    if (isAdjacent) {
      const newTiles = [...tiles];
      [newTiles[index], newTiles[emptyIndex]] = [newTiles[emptyIndex], newTiles[index]];
      setTiles(newTiles);
      setMoves(moves + 1);

      // Check if puzzle is complete
      const isComplete = newTiles.every((tile, index) => 
        tile === null ? index === newTiles.length - 1 : tile === index + 1
      );
      setIsComplete(isComplete);
    }
  };

  return (
    <div className="puzzle-world">
      <motion.button
        className="back-button"
        onClick={() => navigate('/child/games')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Games
      </motion.button>

      <div className="game-header">
        <h1>Puzzle World</h1>
        <div className="game-info">
          <span className="moves">Moves: {moves}</span>
          <button className="reset-btn" onClick={initializePuzzle}>
            Reset Puzzle
          </button>
        </div>
      </div>

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