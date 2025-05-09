import React, { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import './MathSafari.css';

// Emotion-to-background mapping (same as WordWizard)
const emotionBackgrounds = {
  Happiness: 'https://i.pinimg.com/736x/3c/c2/4c/3cc24c1323758ad3ac771422cca85b16.jpg',
  Sadness: 'https://i.pinimg.com/736x/af/a3/93/afa3935151761fafefe50b3b4cf4e22b.jpg',
  Anger: 'https://i.pinimg.com/736x/1b/c2/54/1bc254fc6ac4e9bc66c906b8e222c9e5.jpg',
  Surprise: 'https://i.pinimg.com/736x/b5/08/2c/b5082cfb446b91fde276b51692f61f8b.jpg',
  Disgust: 'https://i.pinimg.com/736x/e3/ed/87/e3ed8733e6a1ff0400821e2c829a11bd.jpg',
  Fear: 'https://i.pinimg.com/736x/86/b6/59/86b659584ccc8d660248fef17e6dad7b.jpg',
  Neutral: 'https://i.pinimg.com/736x/03/98/cb/0398cbb268528dbad35799ad602128be.jpg',
};

const DIFFICULTY_LEVELS = {
  easy: { max: 10, operations: ['+', '-'] },
  medium: { max: 20, operations: ['+', '-', '×'] },
  hard: { max: 50, operations: ['+', '-', '×', '÷'] }
};

const ANIMALS = {
  elephant: '🐘',
  giraffe: '🦒',
  lion: '🦁',
  monkey: '🐒',
  zebra: '🦓',
  hippo: '🦛'
};

const MathSafari = ({ emotion = 'Neutral' }) => {
  const navigate = useNavigate();
  const [score, setScore] = useState(0);
  const [level, setLevel] = useState('easy');
  const [problem, setProblem] = useState(null);
  const [userAnswer, setUserAnswer] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [streak, setStreak] = useState(0);
  const [currentAnimal, setCurrentAnimal] = useState('elephant');
  const [error, setError] = useState(null);

  const generateProblem = useCallback(() => {
    const { max, operations } = DIFFICULTY_LEVELS[level];
    const operation = operations[Math.floor(Math.random() * operations.length)];
    let num1, num2, answer;

    switch (operation) {
      case '+':
        num1 = Math.floor(Math.random() * max);
        num2 = Math.floor(Math.random() * max);
        answer = num1 + num2;
        break;
      case '-':
        num1 = Math.floor(Math.random() * max);
        num2 = Math.floor(Math.random() * (num1 + 1));
        answer = num1 - num2;
        break;
      case '×':
        num1 = Math.floor(Math.random() * (max / 2));
        num2 = Math.floor(Math.random() * (max / 2));
        answer = num1 * num2;
        break;
      case '÷':
        num2 = Math.floor(Math.random() * (max / 4)) + 1;
        answer = Math.floor(Math.random() * (max / 4)) + 1;
        num1 = num2 * answer;
        break;
      default:
        break;
    }

    setProblem({ num1, num2, operation, answer });
    setUserAnswer('');
    setFeedback(null);
  }, [level]);

  // Reset game and clear emotions logs
  const resetGame = async () => {
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

    setScore(0);
    setLevel('easy');
    setStreak(0);
    setCurrentAnimal('elephant');
    generateProblem();
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

  // Initialize game and clear emotions logs on mount
  useEffect(() => {
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

      generateProblem();
    };

    initializeGame();

    // Cleanup: Append percentages when component unmounts
    return () => {
      appendEmotionPercentages();
      console.log('MathSafari unmounted: Emotion percentages appended.');
    };
  }, [generateProblem]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const numAnswer = parseInt(userAnswer);
    
    if (numAnswer === problem.answer) {
      setFeedback({ correct: true, message: 'Great job! 🎉' });
      setScore(prev => prev + (streak + 1) * 10);
      setStreak(prev => prev + 1);
      
      if ((streak + 1) % 5 === 0) {
        const animals = Object.keys(ANIMALS);
        const currentIndex = animals.indexOf(currentAnimal);
        const nextAnimal = animals[(currentIndex + 1) % animals.length];
        setCurrentAnimal(nextAnimal);
      }

      setTimeout(generateProblem, 1500);
    } else {
      setFeedback({ correct: false, message: 'Try again! 💪' });
      setStreak(0);
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
      className="math-safari" 
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
        <h1>Math Safari</h1>
        <div className="game-info">
          <span className="score">Score: {score}</span>
          <span className="streak">Streak: {streak} 🔥</span>
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

      <div className="difficulty-selector">
        {Object.keys(DIFFICULTY_LEVELS).map((diff) => (
          <motion.button
            key={diff}
            className={`difficulty-btn ${level === diff ? 'active' : ''}`}
            onClick={() => setLevel(diff)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {diff.charAt(0).toUpperCase() + diff.slice(1)}
          </motion.button>
        ))}
      </div>

      <motion.div 
        className="problem-container"
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
      >
        <div className="animal-container">
          {ANIMALS[currentAnimal]}
        </div>
        
        {problem && (
          <form onSubmit={handleSubmit} className="problem-form">
            <div className="problem">
              <span>{problem.num1}</span>
              <span>{problem.operation}</span>
              <span>{problem.num2}</span>
              <span>=</span>
              <input
                type="number"
                value={userAnswer}
                onChange={(e) => setUserAnswer(e.target.value)}
                placeholder="?"
                autoFocus
              />
            </div>
            <motion.button
              type="submit"
              className="submit-btn"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Check Answer
            </motion.button>
          </form>
        )}

        {feedback && (
          <motion.div
            className={`feedback ${feedback.correct ? 'correct' : 'incorrect'}`}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
          >
            {feedback.message}
          </motion.div>
        )}
      </motion.div>
    </div>
  );
};

export default MathSafari;