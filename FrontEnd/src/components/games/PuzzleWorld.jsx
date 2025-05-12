import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './PuzzleWorld.css';

const PUZZLE_IMAGES = {
  easy: [
    'https://images.pexels.com/photos/3608263/pexels-photo-3608263.jpeg',
    'https://images.pexels.com/photos/3608542/pexels-photo-3608542.jpeg',
    'https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/57416/cat-sweet-kitty-animals-57416.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/1166473/pexels-photo-1166473.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/31467787/pexels-photo-31467787/free-photo-of-giant-teddy-bear-relaxing-on-park-bench-in-garden.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/31877265/pexels-photo-31877265/free-photo-of-eastern-chipmunk-in-natural-habitat.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/5806696/pexels-photo-5806696.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/462118/pexels-photo-462118.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2'
  ],
  medium: [
   'https://images.pexels.com/photos/60597/dahlia-red-blossom-bloom-60597.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/15580060/pexels-photo-15580060/free-photo-of-deer-fawns-on-meadow.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/10895596/pexels-photo-10895596.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/4001296/pexels-photo-4001296.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/30942994/pexels-photo-30942994/free-photo-of-vibrant-monarch-butterfly-on-orange-zinnia-flower.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/878560/pexels-photo-878560.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/36762/scarlet-honeyeater-bird-red-feathers.jpg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/108370/pexels-photo-108370.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/19036120/pexels-photo-19036120/free-photo-of-delicious-healthy-fruits.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/830829/pexels-photo-830829.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
  ],
  hard: [
    'https://images.pexels.com/photos/994605/pexels-photo-994605.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/1033729/pexels-photo-1033729.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/139099/pexels-photo-139099.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/337909/pexels-photo-337909.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/2196587/pexels-photo-2196587.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/97533/pexels-photo-97533.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/1133957/pexels-photo-1133957.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/236599/pexels-photo-236599.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/30705791/pexels-photo-30705791/free-photo-of-zebra-grazing-in-maasai-mara-national-reserve.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2',
    'https://images.pexels.com/photos/458976/pexels-photo-458976.jpeg'
  ]
};

const LEVELS = {
  easy: Array.from({ length: 10 }, (_, i) => ({
    id: i + 1,
    size: 3,
    image: PUZZLE_IMAGES.easy[i]
  })),
  medium: Array.from({ length: 10 }, (_, i) => ({
    id: i + 1,
    size: 4,
    image: PUZZLE_IMAGES.medium[i]
  })),
  hard: Array.from({ length: 10 }, (_, i) => ({
    id: i + 1,
    size: 5,
    image: PUZZLE_IMAGES.hard[i]
  }))
};

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
  const [currentLevel, setCurrentLevel] = useState('easy');
  const [imageIndex, setImageIndex] = useState(0);
  const [tiles, setTiles] = useState([]);
  const [moves, setMoves] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
<<<<<<< HEAD
  const [gameStarted, setGameStarted] = useState(false);
  const [bestMoves, setBestMoves] = useState({});
=======
  const [error, setError] = useState(null);
>>>>>>> 8216ad021b5d9a845c7c10a11df776a6c738c03b

  const currentPuzzle = LEVELS[currentLevel][imageIndex];

  const isSolvable = (tiles, size) => {
    let inversions = 0;
    const flatTiles = [...tiles];
    
    for (let i = 0; i < flatTiles.length - 1; i++) {
      if (!flatTiles[i]) continue;
      for (let j = i + 1; j < flatTiles.length; j++) {
        if (!flatTiles[j]) continue;
        if (flatTiles[i] > flatTiles[j]) inversions++;
      }
    }

    const emptyRowFromBottom = Math.floor(flatTiles.indexOf(null) / size);
    return (size % 2 === 1) ? 
      (inversions % 2 === 0) : 
      ((inversions + emptyRowFromBottom) % 2 === 0);
  };

  const shuffleTiles = useCallback(() => {
    const size = currentPuzzle.size * currentPuzzle.size;
    let shuffled;
    do {
      shuffled = Array.from({ length: size - 1 }, (_, i) => i + 1);
      shuffled.push(null);
      
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
    } while (!isSolvable(shuffled, currentPuzzle.size));
    
    return shuffled;
  }, [currentPuzzle.size]);

  const initializePuzzle = useCallback(() => {
<<<<<<< HEAD
    setTiles(shuffleTiles());
=======
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
>>>>>>> 8216ad021b5d9a845c7c10a11df776a6c738c03b
    setMoves(0);
    setIsComplete(false);
  }, [shuffleTiles]);

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
<<<<<<< HEAD
  }, [initializePuzzle, currentPuzzle]);
=======

    // Cleanup: Append percentages when component unmounts
    return () => {
      appendEmotionPercentages();
      console.log('PuzzleWorld unmounted: Emotion percentages appended.');
    };
  }, [currentPuzzle, initializePuzzle]);
>>>>>>> 8216ad021b5d9a845c7c10a11df776a6c738c03b

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

<<<<<<< HEAD
      const completed = newTiles.every((tile, index) => 
=======
      const isComplete = newTiles.every((tile, index) => 
>>>>>>> 8216ad021b5d9a845c7c10a11df776a6c738c03b
        tile === null ? index === newTiles.length - 1 : tile === index + 1
      );

      if (completed) {
        const puzzleKey = `${currentLevel}-${imageIndex}`;
        const currentBest = bestMoves[puzzleKey] || Infinity;
        if (moves < currentBest) {
          setBestMoves(prev => ({ ...prev, [puzzleKey]: moves }));
        }
        setIsComplete(true);
      }
    }
  };

<<<<<<< HEAD
  const handleNextPuzzle = () => {
    if (imageIndex < LEVELS[currentLevel].length - 1) {
      setImageIndex(prev => prev + 1);
    } else {
      if (currentLevel === 'easy') {
        setCurrentLevel('medium');
      } else if (currentLevel === 'medium') {
        setCurrentLevel('hard');
      } else {
        alert('🎉 Congratulations! You completed all levels!');
        setGameStarted(false);
        return;
      }
      setImageIndex(0);
    }
    initializePuzzle();
  };

  const InstructionsScreen = () => (
    <div className="instructions-screen">
      <motion.button 
        className="back-button" 
        onClick={() => navigate('/child/games')} 
=======
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
>>>>>>> 8216ad021b5d9a845c7c10a11df776a6c738c03b
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Games
      </motion.button>

      <div className="instructions-content">
        <h1>Puzzle World</h1>
<<<<<<< HEAD
        <h2>How to Play</h2>
        <div className="instruction-steps">
          <div className="step">
            <div className="step-number">1</div>
            <p>Click tiles next to the empty space to move them</p>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <p>Arrange tiles in numerical order</p>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <p>Complete the image puzzle</p>
          </div>
          <div className="step">
            <div className="step-number">4</div>
            <p>Progress through Easy, Medium, and Hard levels</p>
          </div>
=======
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
>>>>>>> 8216ad021b5d9a845c7c10a11df776a6c738c03b
        </div>
        <motion.button 
          className="start-game-btn" 
          onClick={() => setGameStarted(true)} 
          whileTap={{ scale: 0.95 }}
        >
          Start Playing
        </motion.button>
      </div>
    </div>
  );

<<<<<<< HEAD
  const GameScreen = () => (
    <div className="puzzle-world">
      <motion.button 
        className="back-button" 
        onClick={() => setGameStarted(false)} 
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Instructions
      </motion.button>

      <div className="game-header">
        <h1>Puzzle World - {currentLevel.charAt(0).toUpperCase() + currentLevel.slice(1)} Level</h1>
        <div className="game-stats">
          <p>Puzzle: {imageIndex + 1}/{LEVELS[currentLevel].length}</p>
          <p>Moves: {moves}</p>
          <p>Best: {bestMoves[`${currentLevel}-${imageIndex}`] || '-'}</p>
        </div>
        <motion.button 
          className="reset-btn" 
          onClick={initializePuzzle} 
          whileTap={{ scale: 0.95 }}
        >
          Reset Puzzle
        </motion.button>
=======
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
>>>>>>> 8216ad021b5d9a845c7c10a11df776a6c738c03b
      </div>

      <motion.div 
        className="puzzle-container"
        style={{ '--grid-size': currentPuzzle.size }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
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
              backgroundPosition: tile
                ? `${((tile - 1) % currentPuzzle.size) * -100}% ${Math.floor((tile - 1) / currentPuzzle.size) * -100}%`
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
          <h2>🎉 Great job!</h2>
          <p>You solved the puzzle in {moves} moves!</p>
          {bestMoves[`${currentLevel}-${imageIndex}`] === moves && (
            <p>🏆 New Best Score!</p>
          )}
          <motion.button
            className="next-puzzle-btn"
            onClick={handleNextPuzzle}
            whileTap={{ scale: 0.95 }}
          >
            Next Puzzle
          </motion.button>
        </motion.div>
      )}
    </div>
  );

  return gameStarted ? <GameScreen /> : <InstructionsScreen />;
};

export default PuzzleWorld;