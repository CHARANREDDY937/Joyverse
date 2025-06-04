import React from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './GamesDashboard.css';

const GamesDashboard = ({ emotion }) => {
  const navigate = useNavigate();

  const games = [
    {
      id: 1,
      name: 'Word Wizard',
      image: 'https://i.pinimg.com/736x/d0/4c/33/d04c33e8307a8a23059030fe1d1ed252.jpg',
      path: '/child/games/word-wizard'
    },
    {
      id: 2,
      name: 'Math Safari',
      image: 'https://i.pinimg.com/736x/15/4a/bb/154abb667eba021cd75e865d956b39e8.jpg',
      path: '/child/games/math-safari'
    },
    {
      id: 3,
      name: 'Memory Match',
      image: 'https://i.pinimg.com/736x/d8/bb/c8/d8bbc8635fbdaafdce23e3a3306a3f84.jpg',
      path: '/child/games/memory'
    },
    {
      id: 4,
      name: 'Spelling Bee',
      image: 'https://i.pinimg.com/736x/f0/4d/23/f04d235f6d3af0bb1c27856a259ecdb7.jpg',
      path: '/child/games/spelling'
    },
    // {
    //   id: 5,
    //   name: 'Science Quest',
    //   image: 'https://i.pinimg.com/736x/a3/c6/e0/a3c6e0e88875cadaea284c8524f8d989.jpg',
    //   path: '/child/games/science'
    // },
    {
      id: 6,
      name: "Puzzle World",
      image: 'https://i.pinimg.com/736x/88/95/7b/88957bd3d472ea7e417f92a9b24f0cbb.jpg',
      path: '/child/games/puzzle'
    },
    // {
    //   id: 7,
    //   name: 'Reading Race',
    //   image: 'https://i.pinimg.com/736x/99/2e/8c/992e8ca8342db78c99363cb35353670a.jpg',
    //   path: '/child/games/reading'
    // },
    {
      id: 8,
      name: 'Art Studio',
      image: 'https://i.pinimg.com/736x/d0/4c/33/d04c33e8307a8a23059030fe1d1ed252.jpg',
      path: '/child/games/art'
    }
    // {
    //   id: 9,
    //   name: 'Music Maker',
    //   image: 'https://i.pinimg.com/736x/dc/34/58/dc3458a11401388aaf5701b371c05928.jpg',
    //   path: '/child/games/music'
    // }
  ];

  const handleGameClick = async (path) => {
    try {
      // console.log('Resuming emotion logging...');
      // const response = await fetch('http://localhost:8000/resume_emotion_logging', {
      //   method: 'POST',
      // });
      // const data = await response.json();
      // if (data.status !== 'success') {
      //   console.error('Failed to resume emotion logging:', data.message);
      // } else {
      //   console.log('Emotion logging resumed');
      // }
    } catch (error) {
      // console.error('Error resuming emotion logging:', error);
    }
    navigate(path);
  };

  const cardVariants = {
    initial: { scale: 0.8, opacity: 0 },
    animate: { 
      scale: 1, 
      opacity: 1,
      transition: { duration: 0.5 }
    },
    hover: { 
      scale: 1.02,
      y: -5,
      transition: { duration: 0.3 }
    }
  };

  return (
    <div className="games-dashboard">
      <div className="dashboard-header">
        <h1>Learning Adventure Games</h1>
        <p>Choose your favorite game and start learning!</p>
      </div>

      <div className="games-grid">
        {games.map((game, index) => (
          <motion.div
            key={game.id}
            className={`game-card card-${index + 1}`}
            variants={cardVariants}
            initial="initial"
            animate="animate"
            whileHover="hover"
            onClick={() => handleGameClick(game.path)}
            style={{ cursor: 'pointer' }}
          >
            <div className="card-content">
              <div className="game-image">
                <img src={game.image} alt={game.name} />
                <div className="overlay">
                  <div className="game-name-box">
                    <span>{game.name}</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default GamesDashboard;