import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './ScienceQuest.css';

const SCIENCE_QUESTIONS = {
  space: [
    {
      question: "Which planet is known as the Red Planet?",
      options: ["Venus", "Mars", "Jupiter", "Saturn"],
      correct: "Mars",
      fact: "Mars appears red because of iron oxide (rust) on its surface! 🔴",
      image: "https://i.pinimg.com/736x/a3/c6/e0/a3c6e0e88875cadaea284c8524f8d989.jpg"
    },
    {
      question: "What is the closest star to Earth?",
      options: ["Polaris", "Alpha Centauri", "The Sun", "Sirius"],
      correct: "The Sun",
      fact: "The Sun is about 93 million miles away from Earth! ☀️",
      image: "https://i.pinimg.com/736x/bb/c7/b1/bbc7b1b93d49a0a7f222d0c63740bfb9.jpg"
    }
  ],
  animals: [
    {
      question: "Which animal has the longest neck?",
      options: ["Elephant", "Giraffe", "Horse", "Camel"],
      correct: "Giraffe",
      fact: "A giraffe's neck can be up to 8 feet long! 🦒",
      image: "https://i.pinimg.com/736x/15/4a/bb/154abb667eba021cd75e865d956b39e8.jpg"
    },
    {
      question: "What is a group of penguins called?",
      options: ["Herd", "Colony", "Pack", "School"],
      correct: "Colony",
      fact: "Penguins huddle together to stay warm in cold weather! 🐧",
      image: "https://i.pinimg.com/736x/d8/bb/c8/d8bbc8635fbdaafdce23e3a3306a3f84.jpg"
    }
  ],
  nature: [
    {
      question: "What makes thunder?",
      options: ["Rain", "Lightning", "Wind", "Clouds"],
      correct: "Lightning",
      fact: "Lightning heats the air so quickly it creates a shock wave we hear as thunder! ⚡",
      image: "https://i.pinimg.com/736x/f0/4d/23/f04d235f6d3af0bb1c27856a259ecdb7.jpg"
    },
    {
      question: "What do plants need to make their own food?",
      options: ["Sunlight", "Moon", "Stars", "Rain"],
      correct: "Sunlight",
      fact: "Plants use sunlight to make food through a process called photosynthesis! 🌱",
      image: "https://i.pinimg.com/736x/a3/c6/e0/a3c6e0e88875cadaea284c8524f8d989.jpg"
    }
  ]
};

const ScienceQuest = () => {
  const navigate = useNavigate();
  const [category, setCategory] = useState('space');
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [showFact, setShowFact] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [error, setError] = useState(null);

  const currentQuestion = SCIENCE_QUESTIONS[category][currentQuestionIndex];

  const handleAnswerSelect = (answer) => {
    if (selectedAnswer !== null) return; // Prevent multiple selections
    
    setSelectedAnswer(answer);
    
    if (answer === currentQuestion.correct) {
      setScore(prev => prev + 10);
    }

    setTimeout(() => {
      setShowFact(true);
    }, 1000);
  };

  const handleNextQuestion = () => {
    if (currentQuestionIndex < SCIENCE_QUESTIONS[category].length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
    } else {
      setCurrentQuestionIndex(0); // Reset to first question of category
    }
    setSelectedAnswer(null);
    setShowFact(false);
  };

  const resetGame = async () => {
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

    setScore(0);
    setCategory('space');
    setCurrentQuestionIndex(0);
    setSelectedAnswer(null);
    setShowFact(false);
  };

  // Initialize game and clear emotions log on mount
  useEffect(() => {
    const initializeGame = async () => {
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
    };

    initializeGame();

    // Cleanup: Signal external emotion detection system to stop logging (if applicable)
    return () => {
      // Note: Add logic here to stop external emotion detection (e.g., webcam) if controlled by the app.
      // Currently, we assume the external system stops when the game unmounts.
      console.log('ScienceQuest unmounted: Emotion logging should stop.');
    };
  }, []);

  return (
    <div className="science-quest">
      <motion.button
        className="back-button"
        onClick={() => navigate('/child/games')}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        ← Back to Games
      </motion.button>

      <div className="game-header">
        <h1>Science Quest</h1>
        <div className="game-info">
          <span className="score">Score: {score}</span>
          <motion.button
            className="reset-btn"
            onClick={resetGame}
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

      <div className="category-selector">
        {Object.keys(SCIENCE_QUESTIONS).map((cat) => (
          <motion.button
            key={cat}
            className={`category-btn ${category === cat ? 'active' : ''}`}
            onClick={() => {
              setCategory(cat);
              setCurrentQuestionIndex(0);
              setSelectedAnswer(null);
              setShowFact(false);
            }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </motion.button>
        ))}
      </div>

      <motion.div 
        className="question-container"
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <motion.img
          src={currentQuestion.image}
          alt="Science illustration"
          className="question-image"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
        />

        <h2 className="question">{currentQuestion.question}</h2>

        <div className="options-grid">
          {currentQuestion.options.map((option, index) => (
            <motion.button
              key={index}
              className={`option-btn ${
                selectedAnswer === option 
                  ? option === currentQuestion.correct 
                    ? 'correct' 
                    : 'incorrect'
                  : ''
              }`}
              onClick={() => handleAnswerSelect(option)}
              disabled={selectedAnswer !== null}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {option}
            </motion.button>
          ))}
        </div>

        {showFact && (
          <motion.div
            className="fact-container"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
          >
            <p className="fun-fact">{currentQuestion.fact}</p>
            <motion.button
              className="next-btn"
              onClick={handleNextQuestion}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Next Question
            </motion.button>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default ScienceQuest;